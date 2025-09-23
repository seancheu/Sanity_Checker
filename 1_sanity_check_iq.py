#!/usr/bin/env python3
"""
I/Q Data Sanity Checker for RF Signal Processing Pipeline

This script validates I/Q data integrity and quality for downstream RF signal analysis.
It checks for data corruption, determines optimal I/Q convention, calculates image rejection
ratio (IRR), and provides comprehensive statistics for quality assessment.

Example usage:
    python 1_sanity_check_iq.py your.wav --fs_hint 20000000
    python 1_sanity_check_iq.py input.wav --out_dir results --max_samples 1000000
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import soundfile as sf
import scipy.signal as sig

# Configuration constants
DEFAULT_FFT_SIZE: int = 1 << 18  # 262144 samples
MIN_FFT_SIZE: int = 1 << 12     # 4096 samples
MAX_FFT_SIZE: int = 1 << 20     # 1048576 samples
DEFAULT_MAX_SAMPLES: int = 10_000_000  # 10M samples for large file sampling
CLIP_THRESHOLD: float = 0.999   # Clipping detection threshold
EPSILON: float = 1e-12          # Numerical stability constant
DEFAULT_IRR_THRESHOLD: float = 20.0  # Good IRR threshold in dB
DEFAULT_KURTOSIS_THRESHOLD: float = 3.0  # Normal distribution baseline

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SanityCheckError(Exception):
    """Custom exception for sanity check failures."""
    pass


def validate_input_file(file_path: Union[str, Path]) -> Path:
    """
    Validate input WAV file exists and is readable.

    Args:
        file_path: Path to input WAV file

    Returns:
        Validated Path object

    Raises:
        SanityCheckError: If file doesn't exist or isn't readable
    """
    path = Path(file_path)

    if not path.exists():
        raise SanityCheckError(f"Input file does not exist: {path}")

    if not path.is_file():
        raise SanityCheckError(f"Input path is not a file: {path}")

    if not os.access(path, os.R_OK):
        raise SanityCheckError(f"Input file is not readable: {path}")

    return path


def create_output_directory(out_dir: Union[str, Path]) -> Path:
    """
    Create output directory if it doesn't exist.

    Args:
        out_dir: Output directory path

    Returns:
        Validated Path object

    Raises:
        SanityCheckError: If directory cannot be created
    """
    path = Path(out_dir)

    try:
        path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise SanityCheckError(f"Cannot create output directory {path}: {e}")

    if not os.access(path, os.W_OK):
        raise SanityCheckError(f"Output directory is not writable: {path}")

    return path


def load_wav_smart(file_path: Path, max_samples: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Smart WAV file loader with memory-efficient subsampling for large files.

    For large files, uses chunk-based reading and subsampling to maintain memory efficiency
    while ensuring representative sampling across the entire file duration.

    Args:
        file_path: Path to WAV file
        max_samples: Maximum samples to load (None for all)

    Returns:
        Tuple of (samples, sample_rate)

    Raises:
        SanityCheckError: If file cannot be loaded or has invalid format
    """
    try:
        # Get file info first
        info = sf.info(file_path)
        total_frames = info.frames

        logger.info(f"Loading WAV file: {file_path}")
        logger.info(f"File info - Frames: {total_frames:,}, Channels: {info.channels}, "
                   f"Sample rate: {info.samplerate} Hz, Duration: {total_frames/info.samplerate:.2f}s")

        # Determine if we need to sample
        if max_samples and total_frames > max_samples:
            # Calculate sampling strategy
            step = max(1, total_frames // max_samples)
            chunk_size = min(1024 * 1024, total_frames // 10)  # 1M samples or 10% of file
            logger.info(f"Large file detected. Using step={step} sampling with {chunk_size:,} sample chunks")

            # Pre-allocate output array
            expected_samples = total_frames // step
            x_sampled = np.zeros((expected_samples, info.channels), dtype=np.float32)

            samples_written = 0
            samples_read = 0

            # Read file in chunks and subsample
            while samples_read < total_frames and samples_written < expected_samples:
                # Calculate chunk boundaries
                chunk_start = samples_read
                chunk_end = min(samples_read + chunk_size, total_frames)

                # Read chunk
                chunk_data = sf.read(file_path,
                                   start=chunk_start,
                                   stop=chunk_end,
                                   always_2d=True,
                                   dtype='float32')

                if isinstance(chunk_data, tuple):
                    chunk_data = chunk_data[0]  # Extract data if tuple returned

                # Subsample the chunk
                chunk_indices = np.arange(0, len(chunk_data), step)
                subsampled_chunk = chunk_data[chunk_indices]

                # Write to output array
                end_idx = min(samples_written + len(subsampled_chunk), expected_samples)
                actual_chunk_size = end_idx - samples_written
                x_sampled[samples_written:end_idx] = subsampled_chunk[:actual_chunk_size]

                samples_written += actual_chunk_size
                samples_read = chunk_end

                # Log progress for very large files
                if chunk_end % (chunk_size * 5) == 0:
                    progress_pct = 100 * samples_read / total_frames
                    logger.debug(f"Progress: {progress_pct:.1f}% ({samples_read:,}/{total_frames:,} frames)")

            # Trim to actual samples written
            x = x_sampled[:samples_written]
            sr = info.samplerate

            logger.info(f"Loaded {x.shape[0]:,} samples after subsampling (step={step})")
        else:
            # Load entire file for smaller files
            x, sr = sf.read(file_path, always_2d=True, dtype='float32')
            logger.info(f"Loaded {x.shape[0]:,} samples (complete file)")

        # Ensure float32 type
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        return x, float(sr)

    except Exception as e:
        raise SanityCheckError(f"Failed to load WAV file {file_path}: {e}")


def load_wav_file(file_path: Path, max_samples: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Load WAV file with error handling and optional sampling for large files.

    This is a wrapper around load_wav_smart for backward compatibility.

    Args:
        file_path: Path to WAV file
        max_samples: Maximum samples to load (None for all)

    Returns:
        Tuple of (samples, sample_rate)

    Raises:
        SanityCheckError: If file cannot be loaded or has invalid format
    """
    return load_wav_smart(file_path, max_samples)


def calculate_adaptive_fft_size(num_samples: int, target_size: int = DEFAULT_FFT_SIZE) -> int:
    """
    Calculate adaptive FFT size based on available samples.

    Args:
        num_samples: Number of available samples
        target_size: Target FFT size

    Returns:
        Optimal FFT size (power of 2)
    """
    # Limit to available samples
    max_size = min(target_size, num_samples)

    # Find largest power of 2 <= max_size
    fft_size = 1 << (max_size.bit_length() - 1)

    # Ensure minimum size
    fft_size = max(MIN_FFT_SIZE, min(MAX_FFT_SIZE, fft_size))

    logger.debug(f"Adaptive FFT size: {fft_size} (target: {target_size}, samples: {num_samples})")

    return fft_size


def calculate_image_rejection_ratio(complex_signal: np.ndarray, fft_size: Optional[int] = None) -> float:
    """
    Calculate image rejection ratio (IRR) in dB.

    The IRR measures the quality of I/Q data by comparing power in positive
    vs negative frequencies. Higher IRR indicates better I/Q balance.

    Args:
        complex_signal: Complex I/Q signal
        fft_size: FFT size for analysis (None for adaptive)

    Returns:
        Image rejection ratio in dB (absolute value)

    Raises:
        SanityCheckError: If calculation fails
    """
    try:
        if len(complex_signal) == 0:
            raise ValueError("Empty complex signal")

        # Use adaptive FFT size if not specified
        if fft_size is None:
            fft_size = calculate_adaptive_fft_size(len(complex_signal))

        # Ensure we don't exceed signal length
        fft_size = min(fft_size, len(complex_signal))

        # Take center portion of signal for analysis
        if len(complex_signal) > fft_size:
            start_idx = (len(complex_signal) - fft_size) // 2
            signal_segment = complex_signal[start_idx:start_idx + fft_size]
        else:
            signal_segment = complex_signal

        # Compute FFT and shift to center DC
        X = np.fft.fftshift(np.fft.fft(signal_segment, n=fft_size))

        # Calculate power spectrum with numerical stability
        P = (np.abs(X) ** 2).astype(np.float64)

        # Split into positive and negative frequency bins
        mid = P.size // 2
        pos_power = P[mid + 1:].sum() + EPSILON  # Positive frequencies
        neg_power = P[:mid].sum() + EPSILON      # Negative frequencies

        # Calculate IRR in dB
        irr_db = 10.0 * np.log10(pos_power / neg_power)

        # Return absolute value (we care about imbalance magnitude)
        return float(abs(irr_db))

    except Exception as e:
        raise SanityCheckError(f"Failed to calculate IRR: {e}")


def calculate_basic_statistics(signal: np.ndarray, name: str) -> Dict[str, float]:
    """
    Calculate basic statistical measures for a signal.

    Args:
        signal: Input signal (real-valued)
        name: Prefix for statistic names

    Returns:
        Dictionary of statistics
    """
    if len(signal) == 0:
        return {f"{name}_{key}": 0.0 for key in ["mean", "std", "peak_abs", "clip_pct"]}

    # Use float64 for intermediate calculations to avoid precision loss
    signal_f64 = signal.astype(np.float64)

    return {
        f"{name}_mean": float(np.mean(signal_f64)),
        f"{name}_std": float(np.std(signal_f64)),
        f"{name}_peak_abs": float(np.max(np.abs(signal_f64))),
        f"{name}_clip_pct": float(100.0 * np.mean(np.abs(signal_f64) > CLIP_THRESHOLD)),
        f"{name}_min": float(np.min(signal_f64)),
        f"{name}_max": float(np.max(signal_f64)),
        f"{name}_rms": float(np.sqrt(np.mean(signal_f64 ** 2)))
    }


def calculate_envelope_kurtosis(complex_signal: np.ndarray) -> float:
    """
    Calculate envelope kurtosis to assess signal variability.

    Kurtosis indicates whether the envelope is constant (low kurtosis)
    or highly variable (high kurtosis), which helps identify signal types.

    Args:
        complex_signal: Complex I/Q signal

    Returns:
        Envelope kurtosis value
    """
    try:
        # Calculate envelope (magnitude)
        envelope = np.abs(complex_signal)

        if len(envelope) == 0:
            return 0.0

        # Normalize envelope
        env_mean = np.mean(envelope)
        env_std = np.std(envelope)

        if env_std < EPSILON:
            return 0.0  # Constant envelope

        normalized_env = (envelope - env_mean) / env_std

        # Calculate kurtosis (4th moment)
        kurtosis = float(np.mean(normalized_env ** 4))

        return kurtosis

    except Exception:
        logger.warning("Failed to calculate envelope kurtosis")
        return 0.0


def determine_optimal_iq_convention(i_channel: np.ndarray, q_channel: np.ndarray,
                                  fft_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Determine optimal I/Q convention by testing different combinations.

    Tests three conventions:
    - I + jQ (standard)
    - I - jQ (Q inverted)
    - Q + jI (channels swapped)

    Args:
        i_channel: I channel data
        q_channel: Q channel data
        fft_size: FFT size for IRR calculation

    Returns:
        Dictionary with IRR results and best convention
    """
    logger.info("Testing I/Q conventions...")

    # Remove DC for more accurate IRR measurement
    i_dc_removed = i_channel - np.mean(i_channel)
    q_dc_removed = q_channel - np.mean(q_channel)

    # Test three conventions
    conventions = {
        "I+jQ": i_dc_removed + 1j * q_dc_removed,
        "I-jQ": i_dc_removed - 1j * q_dc_removed,
        "Q+jI": q_dc_removed + 1j * i_dc_removed
    }

    irr_results = {}
    for name, complex_signal in conventions.items():
        try:
            irr_db = calculate_image_rejection_ratio(complex_signal, fft_size)
            irr_results[f"irr_db_{name.replace('+j', '_').replace('-j', '_neg_').replace('j', '')}"] = irr_db
            logger.debug(f"IRR for {name}: {irr_db:.2f} dB")
        except Exception as e:
            logger.warning(f"Failed to calculate IRR for {name}: {e}")
            irr_results[f"irr_db_{name.replace('+j', '_').replace('-j', '_neg_').replace('j', '')}"] = 0.0

    # Find best convention
    best_irr = 0.0
    best_convention = "I+jQ"

    for key, irr in irr_results.items():
        if irr > best_irr:
            best_irr = irr
            if "I_Q" in key:
                best_convention = "I+jQ"
            elif "I_neg_Q" in key:
                best_convention = "I-jQ"
            elif "Q_I" in key:
                best_convention = "Q+jI"

    logger.info(f"Best I/Q convention: {best_convention} (IRR: {best_irr:.2f} dB)")

    return {
        **irr_results,
        "best_convention": best_convention,
        "best_irr_db": best_irr
    }


def generate_quality_assessment(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate quality assessment based on calculated statistics.

    Args:
        stats: Dictionary of calculated statistics

    Returns:
        Quality assessment dictionary
    """
    assessment = {
        "quality_score": 0.0,
        "warnings": [],
        "recommendations": [],
        "issues": []
    }

    score = 100.0  # Start with perfect score

    # Check for data corruption
    if stats.get("nan_count", 0) > 0:
        assessment["issues"].append(f"Found {stats['nan_count']} NaN values")
        score -= 20

    if stats.get("inf_count", 0) > 0:
        assessment["issues"].append(f"Found {stats['inf_count']} infinite values")
        score -= 20

    # Check for clipping
    i_clip = stats.get("I_clip_pct", 0)
    q_clip = stats.get("Q_clip_pct", 0)
    max_clip = max(i_clip, q_clip)

    if max_clip > 5.0:
        assessment["issues"].append(f"Severe clipping detected: {max_clip:.1f}%")
        score -= 25
    elif max_clip > 1.0:
        assessment["warnings"].append(f"Moderate clipping detected: {max_clip:.1f}%")
        score -= 10
    elif max_clip > 0.1:
        assessment["warnings"].append(f"Minor clipping detected: {max_clip:.1f}%")
        score -= 5

    # Check IRR quality
    best_irr = stats.get("best_irr_db", 0)
    if best_irr < 10:
        assessment["issues"].append(f"Poor I/Q balance (IRR: {best_irr:.1f} dB)")
        score -= 15
    elif best_irr < DEFAULT_IRR_THRESHOLD:
        assessment["warnings"].append(f"Suboptimal I/Q balance (IRR: {best_irr:.1f} dB)")
        score -= 5

    # Check I/Q convention
    best_conv = stats.get("best_convention", "I+jQ")
    if best_conv != "I+jQ":
        assessment["recommendations"].append(f"Consider using --conv {best_conv.replace('+j', ' ').replace('-j', ' -').replace('j', '')} in downstream processing")

    # Check envelope characteristics
    kurtosis = stats.get("envelope_kurtosis", 3.0)
    if kurtosis < 1.5:
        assessment["warnings"].append("Very constant envelope detected - may indicate CW or unmodulated carrier")
    elif kurtosis > 10:
        assessment["warnings"].append("Highly variable envelope detected - check for intermittent signals or interference")

    # Check duration
    duration = stats.get("duration_s", 0)
    if duration < 0.1:
        assessment["warnings"].append(f"Very short recording ({duration:.3f}s) - may not be sufficient for analysis")

    # Ensure score doesn't go negative
    assessment["quality_score"] = max(0.0, score)

    return assessment


def generate_output_files(stats: Dict[str, Any], assessment: Dict[str, Any],
                         input_file: Path, output_dir: Path) -> Tuple[Path, Path]:
    """
    Generate output files with statistics and human-readable report.

    Args:
        stats: Statistics dictionary
        assessment: Quality assessment dictionary
        input_file: Original input file path
        output_dir: Output directory

    Returns:
        Tuple of (stats_file_path, report_file_path)
    """
    base_name = input_file.stem

    # Generate output filenames
    stats_file = output_dir / f"sanity_check_{base_name}_stats.json"
    report_file = output_dir / f"sanity_check_{base_name}_report.txt"

    # Write JSON statistics
    try:
        with open(stats_file, 'w') as f:
            json.dump({**stats, "quality_assessment": assessment}, f, indent=2, sort_keys=True)
        logger.info(f"Statistics written to: {stats_file}")
    except Exception as e:
        logger.error(f"Failed to write statistics file: {e}")
        raise SanityCheckError(f"Cannot write statistics to {stats_file}: {e}")

    # Write human-readable report
    try:
        with open(report_file, 'w') as f:
            f.write("I/Q Data Sanity Check Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Input File: {input_file}\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Quality Score: {assessment['quality_score']:.1f}/100\n\n")

            # File characteristics
            f.write("File Characteristics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Duration: {stats['duration_s']:.3f} seconds\n")
            f.write(f"Samples: {stats['frames']:,}\n")
            f.write(f"Channels: {stats['channels']}\n")
            f.write(f"Sample Rate (header): {stats['fs_header_hz']:,.0f} Hz\n")
            f.write(f"Sample Rate (used): {stats['fs_used_hz']:,.0f} Hz\n\n")

            # Data quality
            f.write("Data Quality:\n")
            f.write("-" * 20 + "\n")
            f.write(f"NaN values: {stats['nan_count']}\n")
            f.write(f"Infinite values: {stats['inf_count']}\n")
            f.write(f"I channel clipping: {stats.get('I_clip_pct', 0):.2f}%\n")
            f.write(f"Q channel clipping: {stats.get('Q_clip_pct', 0):.2f}%\n\n")

            # I/Q balance
            f.write("I/Q Balance Analysis:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Best convention: {stats['best_convention']}\n")
            f.write(f"Best IRR: {stats['best_irr_db']:.2f} dB\n")
            f.write(f"Envelope kurtosis: {stats['envelope_kurtosis']:.2f}\n\n")

            # Issues and recommendations
            if assessment['issues']:
                f.write("Issues Found:\n")
                f.write("-" * 20 + "\n")
                for issue in assessment['issues']:
                    f.write(f"[ERROR] {issue}\n")
                f.write("\n")

            if assessment['warnings']:
                f.write("Warnings:\n")
                f.write("-" * 20 + "\n")
                for warning in assessment['warnings']:
                    f.write(f"⚠️ {warning}\n")
                f.write("\n")

            if assessment['recommendations']:
                f.write("Recommendations:\n")
                f.write("-" * 20 + "\n")
                for rec in assessment['recommendations']:
                    f.write(f"[SUGGESTION] {rec}\n")
                f.write("\n")

            if not assessment['issues'] and not assessment['warnings']:
                f.write("[OK] No significant issues detected!\n\n")

        logger.info(f"Report written to: {report_file}")
    except Exception as e:
        logger.error(f"Failed to write report file: {e}")
        raise SanityCheckError(f"Cannot write report to {report_file}: {e}")

    return stats_file, report_file


def perform_sanity_check(wav_file: Path, fs_hint: Optional[float] = None,
                        fft_size: Optional[int] = None, max_samples: Optional[int] = None,
                        output_dir: Path = Path("out_report")) -> Dict[str, Any]:
    """
    Perform comprehensive I/Q data sanity check.

    Args:
        wav_file: Path to input WAV file
        fs_hint: Override sample rate (Hz)
        fft_size: FFT size for IRR analysis
        max_samples: Maximum samples to load for large files
        output_dir: Output directory for results

    Returns:
        Complete statistics dictionary

    Raises:
        SanityCheckError: If analysis fails
    """
    logger.info(f"Starting sanity check for: {wav_file}")
    start_time = time.time()

    # Load WAV file
    x, sr_header = load_wav_file(wav_file, max_samples)
    N, C = x.shape

    # Determine sample rate to use
    fs = fs_hint if fs_hint is not None else sr_header

    # Validate channel count
    if C < 2:
        raise SanityCheckError(f"Need 2 channels (I,Q), got {C}")

    logger.info(f"Processing {N:,} samples at {fs:,.0f} Hz ({N/fs:.3f} seconds)")

    # Extract I and Q channels
    I = x[:, 0]
    Q = x[:, 1]

    # Initialize statistics dictionary
    stats = {
        "analysis_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "input_file": str(wav_file),
        "frames": int(N),
        "channels": int(C),
        "fs_header_hz": float(sr_header),
        "fs_used_hz": float(fs),
        "duration_s": float(N / fs),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "analysis_params": {
            "fft_size": fft_size or calculate_adaptive_fft_size(N),
            "max_samples_loaded": max_samples or N,
            "actual_samples_loaded": N
        }
    }

    # Calculate basic statistics for both channels
    logger.info("Calculating channel statistics...")
    stats.update(calculate_basic_statistics(I, "I"))
    stats.update(calculate_basic_statistics(Q, "Q"))

    # Determine optimal I/Q convention
    logger.info("Analyzing I/Q conventions...")
    iq_analysis = determine_optimal_iq_convention(I, Q, fft_size)
    stats.update(iq_analysis)

    # Calculate envelope kurtosis using best convention
    if iq_analysis["best_convention"] == "I+jQ":
        best_complex = (I - np.mean(I)) + 1j * (Q - np.mean(Q))
    elif iq_analysis["best_convention"] == "I-jQ":
        best_complex = (I - np.mean(I)) - 1j * (Q - np.mean(Q))
    else:  # Q+jI
        best_complex = (Q - np.mean(Q)) + 1j * (I - np.mean(I))

    logger.info("Calculating envelope characteristics...")
    stats["envelope_kurtosis"] = calculate_envelope_kurtosis(best_complex)

    # Generate quality assessment
    logger.info("Generating quality assessment...")
    assessment = generate_quality_assessment(stats)

    elapsed_time = time.time() - start_time
    stats["analysis_duration_s"] = elapsed_time

    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    logger.info(f"Quality score: {assessment['quality_score']:.1f}/100")

    return {**stats, "quality_assessment": assessment}


def main():
    """Main entry point for the sanity check script."""
    parser = argparse.ArgumentParser(
        description="I/Q Data Sanity Checker for RF Signal Processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "wav",
        type=str,
        help="Input WAV file containing I/Q data"
    )

    # Optional arguments
    parser.add_argument(
        "--fs_hint",
        type=float,
        default=None,
        help="Override sample rate from WAV header (Hz)"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="out_report",
        help="Output directory for results"
    )

    parser.add_argument(
        "--fft_size",
        type=int,
        default=None,
        help=f"FFT size for IRR analysis (default: adaptive, max {MAX_FFT_SIZE})"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Maximum samples to load for large files (None for all)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (errors only)"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    try:
        # Validate inputs
        input_file = validate_input_file(args.wav)
        output_dir = create_output_directory(args.out_dir)

        # Validate FFT size if provided
        if args.fft_size is not None:
            if args.fft_size < MIN_FFT_SIZE or args.fft_size > MAX_FFT_SIZE:
                raise SanityCheckError(f"FFT size must be between {MIN_FFT_SIZE} and {MAX_FFT_SIZE}")
            if args.fft_size & (args.fft_size - 1) != 0:
                raise SanityCheckError("FFT size must be a power of 2")

        # Perform sanity check
        results = perform_sanity_check(
            wav_file=input_file,
            fs_hint=args.fs_hint,
            fft_size=args.fft_size,
            max_samples=args.max_samples if args.max_samples > 0 else None,
            output_dir=output_dir
        )

        # Generate output files
        stats_file, report_file = generate_output_files(
            stats=results,
            assessment=results["quality_assessment"],
            input_file=input_file,
            output_dir=output_dir
        )

        # Print summary to stdout (for backward compatibility)
        if not args.quiet:
            print(json.dumps(results, indent=2))

        # Print final status
        quality_score = results["quality_assessment"]["quality_score"]
        if quality_score >= 80:
            logger.info("[OK] I/Q data quality is good")
        elif quality_score >= 60:
            logger.warning("[WARNING] I/Q data quality is marginal")
        else:
            logger.error("[ERROR] I/Q data quality is poor")

        return 0

    except SanityCheckError as e:
        logger.error(f"Sanity check failed: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

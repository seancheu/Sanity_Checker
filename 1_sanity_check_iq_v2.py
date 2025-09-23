#!/usr/bin/env python3
"""
I/Q Data Sanity Checker for RF Signal Processing Pipeline

Improvements:
- Per-run unique output subfolder under --out_dir
- UTF-8 file writes
- Multi-window analysis: analyze representative slices across huge files
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List

import numpy as np
import soundfile as sf
import scipy.signal as sig  # imported for future use

# Configuration constants
DEFAULT_FFT_SIZE: int = 1 << 18   # 262144 samples
MIN_FFT_SIZE: int = 1 << 12       # 4096 samples
MAX_FFT_SIZE: int = 1 << 20       # 1048576 samples
DEFAULT_MAX_SAMPLES: int = 10_000_000
CLIP_THRESHOLD: float = 0.999
EPSILON: float = 1e-12
DEFAULT_IRR_THRESHOLD: float = 20.0
DEFAULT_KURTOSIS_THRESHOLD: float = 3.0

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SanityCheckError(Exception):
    pass


def validate_input_file(file_path: Union[str, Path]) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise SanityCheckError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise SanityCheckError(f"Input path is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise SanityCheckError(f"Input file is not readable: {path}")
    return path


def create_output_directory(out_dir: Union[str, Path]) -> Path:
    path = Path(out_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise SanityCheckError(f"Cannot create output directory {path}: {e}")
    if not os.access(path, os.W_OK):
        raise SanityCheckError(f"Output directory is not writable: {path}")
    return path


def create_unique_run_directory(base_out_dir: Union[str, Path], input_file: Path) -> Path:
    base = create_output_directory(base_out_dir)
    ts = time.strftime('%Y%m%d_%H%M%S')
    stem = input_file.stem
    candidate = base / f"run_{stem}_{ts}"
    idx = 1
    while candidate.exists():
        candidate = base / f"run_{stem}_{ts}_{idx:02d}"
        idx += 1
    try:
        candidate.mkdir(parents=True, exist_ok=False)
    except (OSError, PermissionError) as e:
        raise SanityCheckError(f"Cannot create run directory {candidate}: {e}")
    return candidate


def load_wav_smart(file_path: Path, max_samples: Optional[int] = None) -> Tuple[np.ndarray, float, int]:
    """
    Legacy subsampling loader (kept for compatibility). Returns (samples, sr, total_frames).
    """
    try:
        info = sf.info(file_path)
        total_frames = info.frames

        logger.info(f"Loading WAV file: {file_path}")
        logger.info(f"File info - Frames: {total_frames:,}, Channels: {info.channels}, "
                    f"Sample rate: {info.samplerate} Hz, Duration: {total_frames/info.samplerate:.2f}s")

        if max_samples and total_frames > max_samples:
            step = max(1, total_frames // max_samples)
            chunk_size = min(1024 * 1024, total_frames // 10)
            logger.info(f"Large file detected. Using step={step} sampling with {chunk_size:,} sample chunks")

            expected_samples = total_frames // step
            x_sampled = np.zeros((expected_samples, info.channels), dtype=np.float32)

            samples_written = 0
            samples_read = 0

            while samples_read < total_frames and samples_written < expected_samples:
                chunk_start = samples_read
                chunk_end = min(samples_read + chunk_size, total_frames)

                chunk_data = sf.read(file_path, start=chunk_start, stop=chunk_end, always_2d=True, dtype='float32')
                if isinstance(chunk_data, tuple):
                    chunk_data = chunk_data[0]

                chunk_indices = np.arange(0, len(chunk_data), step)
                subsampled_chunk = chunk_data[chunk_indices]

                end_idx = min(samples_written + len(subsampled_chunk), expected_samples)
                actual_chunk_size = end_idx - samples_written
                x_sampled[samples_written:end_idx] = subsampled_chunk[:actual_chunk_size]

                samples_written += actual_chunk_size
                samples_read = chunk_end

                if chunk_end % (chunk_size * 5) == 0:
                    progress_pct = 100 * samples_read / total_frames
                    logger.debug(f"Progress: {progress_pct:.1f}% ({samples_read:,}/{total_frames:,} frames)")

            x = x_sampled[:samples_written]
            sr = info.samplerate
            logger.info(f"Loaded {x.shape[0]:,} samples after subsampling (step={step})")
        else:
            x, sr = sf.read(file_path, always_2d=True, dtype='float32')
            logger.info(f"Loaded {x.shape[0]:,} samples (complete file)")

        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        return x, float(sr), int(total_frames)

    except Exception as e:
        raise SanityCheckError(f"Failed to load WAV file {file_path}: {e}")


def load_wav_windows(file_path: Path, n_windows: int, window_size: int) -> Tuple[np.ndarray, float, int]:
    """
    Evenly-spaced window loader for huge files. Reads n_windows non-overlapping
    segments of length window_size, spread across the file. Returns concatenated
    array of all windows [sum_len, channels], sample_rate, total_frames.

    This keeps memory bounded and preserves representativeness across time.
    """
    try:
        info = sf.info(file_path)
        total_frames = info.frames
        sr = info.samplerate
        channels = info.channels

        logger.info(f"Loading {n_windows} windows × {window_size} from: {file_path}")
        logger.info(f"File info - Frames: {total_frames:,}, Channels: {channels}, "
                    f"Sample rate: {sr} Hz, Duration: {total_frames/sr:.2f}s")

        if n_windows <= 0 or window_size <= 0:
            raise SanityCheckError("n_windows and window_size must be positive when using windowed analysis")

        # Compute start positions evenly spaced over [0, total_frames - window_size]
        usable = max(0, total_frames - window_size)
        if n_windows == 1:
            starts = [usable // 2]
        else:
            starts = np.linspace(0, usable, num=n_windows, dtype=np.int64).tolist()

        out = np.zeros((n_windows * window_size, channels), dtype=np.float32)
        write_pos = 0

        for i, s in enumerate(starts):
            e = int(s + window_size)
            data = sf.read(file_path, start=int(s), stop=int(e), always_2d=True, dtype='float32')
            if isinstance(data, tuple):
                data = data[0]
            # If short at EOF, pad with zeros (rare)
            if data.shape[0] < window_size:
                tmp = np.zeros((window_size, channels), dtype=np.float32)
                tmp[:data.shape[0]] = data
                data = tmp
            out[write_pos:write_pos + window_size] = data
            write_pos += window_size

        return out, float(sr), int(total_frames)

    except Exception as e:
        raise SanityCheckError(f"Failed to load windows from WAV file {file_path}: {e}")


def calculate_adaptive_fft_size(num_samples: int, target_size: int = DEFAULT_FFT_SIZE) -> int:
    max_size = min(target_size, num_samples)
    fft_size = 1 << (max_size.bit_length() - 1)
    fft_size = max(MIN_FFT_SIZE, min(MAX_FFT_SIZE, fft_size))
    logger.debug(f"Adaptive FFT size: {fft_size} (target: {target_size}, samples: {num_samples})")
    return fft_size


def calculate_image_rejection_ratio(complex_signal: np.ndarray, fft_size: Optional[int] = None) -> float:
    try:
        if len(complex_signal) == 0:
            raise ValueError("Empty complex signal")
        if fft_size is None:
            fft_size = calculate_adaptive_fft_size(len(complex_signal))
        fft_size = min(fft_size, len(complex_signal))
        if len(complex_signal) > fft_size:
            start_idx = (len(complex_signal) - fft_size) // 2
            signal_segment = complex_signal[start_idx:start_idx + fft_size]
        else:
            signal_segment = complex_signal
        X = np.fft.fftshift(np.fft.fft(signal_segment, n=fft_size))
        P = (np.abs(X) ** 2).astype(np.float64)
        mid = P.size // 2
        pos_power = P[mid + 1:].sum() + EPSILON
        neg_power = P[:mid].sum() + EPSILON
        irr_db = 10.0 * np.log10(pos_power / neg_power)
        return float(abs(irr_db))
    except Exception as e:
        raise SanityCheckError(f"Failed to calculate IRR: {e}")


def calculate_basic_statistics(signal: np.ndarray, name: str) -> Dict[str, float]:
    if len(signal) == 0:
        return {f"{name}_{key}": 0.0 for key in ["mean", "std", "peak_abs", "clip_pct"]}
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
    try:
        envelope = np.abs(complex_signal)
        if len(envelope) == 0:
            return 0.0
        env_mean = np.mean(envelope)
        env_std = np.std(envelope)
        if env_std < EPSILON:
            return 0.0
        normalized_env = (envelope - env_mean) / env_std
        kurtosis = float(np.mean(normalized_env ** 4))
        return kurtosis
    except Exception:
        logger.warning("Failed to calculate envelope kurtosis")
        return 0.0


def determine_optimal_iq_convention(i_channel: np.ndarray, q_channel: np.ndarray,
                                    fft_size: Optional[int] = None) -> Dict[str, Any]:
    logger.info("Testing I/Q conventions...")
    i_dc_removed = i_channel - np.mean(i_channel)
    q_dc_removed = q_channel - np.mean(q_channel)
    conventions = {
        "I+jQ": i_dc_removed + 1j * q_dc_removed,
        "I-jQ": i_dc_removed - 1j * q_dc_removed,
        "Q+jI": q_dc_removed + 1j * i_dc_removed
    }
    irr_results = {}
    for name, complex_signal in conventions.items():
        try:
            irr_db = calculate_image_rejection_ratio(complex_signal, fft_size)
            key = f"irr_db_{name.replace('+j', '_').replace('-j', '_neg_').replace('j', '')}"
            irr_results[key] = irr_db
        except Exception as e:
            logger.warning(f"Failed to calculate IRR for {name}: {e}")
            key = f"irr_db_{name.replace('+j', '_').replace('-j', '_neg_').replace('j', '')}"
            irr_results[key] = 0.0
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
    return {**irr_results, "best_convention": best_convention, "best_irr_db": best_irr}


def generate_quality_assessment(stats: Dict[str, Any]) -> Dict[str, Any]:
    assessment = {"quality_score": 0.0, "warnings": [], "recommendations": [], "issues": []}
    score = 100.0
    if stats.get("nan_count", 0) > 0:
        assessment["issues"].append(f"Found {stats['nan_count']} NaN values")
        score -= 20
    if stats.get("inf_count", 0) > 0:
        assessment["issues"].append(f"Found {stats['inf_count']} infinite values")
        score -= 20
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
    best_irr = stats.get("best_irr_db", 0)
    if best_irr < 10:
        assessment["issues"].append(f"Poor I/Q balance (IRR: {best_irr:.1f} dB)")
        score -= 15
    elif best_irr < DEFAULT_IRR_THRESHOLD:
        assessment["warnings"].append(f"Suboptimal I/Q balance (IRR: {best_irr:.1f} dB)")
        score -= 5
    best_conv = stats.get("best_convention", "I+jQ")
    if best_conv != "I+jQ":
        assessment["recommendations"].append(
            f"Consider using --conv {best_conv.replace('+j', ' ').replace('-j', ' -').replace('j', '')} in downstream processing"
        )
    kurtosis = stats.get("envelope_kurtosis", 3.0)
    if kurtosis < 1.5:
        assessment["warnings"].append("Very constant envelope detected - may indicate CW or unmodulated carrier")
    elif kurtosis > 10:
        assessment["warnings"].append("Highly variable envelope detected - check for intermittent signals or interference")
    duration = stats.get("duration_s", 0)
    if duration < 0.1:
        assessment["warnings"].append(f"Very short recording ({duration:.3f}s) - may not be sufficient for analysis")
    assessment["quality_score"] = max(0.0, score)
    return assessment


def generate_output_files(stats: Dict[str, Any], assessment: Dict[str, Any],
                          input_file: Path, output_dir: Path) -> Tuple[Path, Path]:
    base_name = input_file.stem
    stats_file = output_dir / f"sanity_check_{base_name}_stats.json"
    report_file = output_dir / f"sanity_check_{base_name}_report.txt"
    try:
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({**stats, "quality_assessment": assessment}, f, indent=2, sort_keys=True, ensure_ascii=False)
        logger.info(f"Statistics written to: {stats_file}")
    except Exception as e:
        logger.error(f"Failed to write statistics file: {e}")
        raise SanityCheckError(f"Cannot write statistics to {stats_file}: {e}")
    try:
        with open(report_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write("I/Q Data Sanity Check Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input File: {input_file}\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Quality Score: {assessment['quality_score']:.1f}/100\n\n")

            f.write("File Characteristics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"File duration: {stats['file_duration_s']:.3f} seconds\n")
            f.write(f"Covered duration (windows total): {stats['covered_duration_s']:.3f} seconds\n")
            f.write(f"Coverage fraction: {stats['coverage_fraction']*100:.4f}%\n")
            f.write(f"Samples analyzed: {stats['frames']:,}\n")
            f.write(f"Channels: {stats['channels']}\n")
            f.write(f"Sample Rate (header): {stats['fs_header_hz']:,.0f} Hz\n")
            f.write(f"Sample Rate (used): {stats['fs_used_hz']:,.0f} Hz\n")
            f.write(f"Sampling strategy: {stats['sampling_strategy']}\n\n")

            f.write("Data Quality:\n")
            f.write("-" * 20 + "\n")
            f.write(f"NaN values: {stats['nan_count']}\n")
            f.write(f"Infinite values: {stats['inf_count']}\n")
            f.write(f"I channel clipping: {stats.get('I_clip_pct', 0):.2f}%\n")
            f.write(f"Q channel clipping: {stats.get('Q_clip_pct', 0):.2f}%\n\n")

            f.write("I/Q Balance Analysis:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Best convention: {stats['best_convention']}\n")
            f.write(f"Best IRR: {stats['best_irr_db']:.2f} dB\n")
            f.write(f"Envelope kurtosis: {stats['envelope_kurtosis']:.2f}\n\n")

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
                         output_dir: Path = Path("out_report"),
                         n_windows: int = 16, window_size: int = DEFAULT_FFT_SIZE) -> Dict[str, Any]:
    """
    Perform I/Q sanity check using either multi-window or legacy subsampling.
    """
    logger.info(f"Starting sanity check for: {wav_file}")
    start_time = time.time()

    # Probe file
    info = sf.info(wav_file)
    total_frames_header = int(info.frames)
    sr_header = float(info.samplerate)
    fs = fs_hint if fs_hint is not None else sr_header
    file_duration_s = total_frames_header / fs

    # Choose loading strategy
    if n_windows and n_windows > 0:
        x, sr_loaded, total_frames_reported = load_wav_windows(wav_file, n_windows=n_windows, window_size=window_size)
        sampling_strategy = f"even_windows(n={n_windows}, window_size={window_size})"
        covered_frames = x.shape[0]
    else:
        x, sr_loaded, total_frames_reported = load_wav_smart(wav_file, max_samples=max_samples)
        sampling_strategy = f"legacy_subsample(max_samples={max_samples})" if max_samples else "full_load"
        covered_frames = x.shape[0]

    if abs(sr_loaded - fs) > 1e-6:
        logger.warning(f"Using fs_hint {fs:.0f} Hz different from header {sr_loaded:.0f} Hz.")

    N, C = x.shape
    if C < 2:
        raise SanityCheckError(f"Need 2 channels (I,Q), got {C}")
    logger.info(f"Processing {N:,} samples at {fs:,.0f} Hz (~{N/fs:.3f} seconds covered)")

    I = x[:, 0]
    Q = x[:, 1]

    # Stats container
    stats = {
        "analysis_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "input_file": str(wav_file),
        "frames": int(N),
        "channels": int(C),
        "fs_header_hz": float(sr_header),
        "fs_used_hz": float(fs),
        "file_duration_s": float(file_duration_s),
        "covered_duration_s": float(N / fs),
        "coverage_fraction": float((N / fs) / file_duration_s if file_duration_s > 0 else 0.0),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "sampling_strategy": sampling_strategy,
        "analysis_params": {
            "fft_size": fft_size or calculate_adaptive_fft_size(N),
            "max_samples_loaded": max_samples if (n_windows is None or n_windows <= 0) else n_windows * window_size,
            "actual_samples_loaded": N,
            "n_windows": n_windows,
            "window_size": window_size
        }
    }

    # Channel stats
    logger.info("Calculating channel statistics...")
    stats.update(calculate_basic_statistics(I, "I"))
    stats.update(calculate_basic_statistics(Q, "Q"))

    # IQ convention / IRR
    logger.info("Analyzing I/Q conventions...")
    iq_analysis = determine_optimal_iq_convention(I, Q, fft_size)
    stats.update(iq_analysis)

    # Envelope kurtosis on best convention
    if iq_analysis["best_convention"] == "I+jQ":
        best_complex = (I - np.mean(I)) + 1j * (Q - np.mean(Q))
    elif iq_analysis["best_convention"] == "I-jQ":
        best_complex = (I - np.mean(I)) - 1j * (Q - np.mean(Q))
    else:
        best_complex = (Q - np.mean(Q)) + 1j * (I - np.mean(I))
    logger.info("Calculating envelope characteristics...")
    stats["envelope_kurtosis"] = calculate_envelope_kurtosis(best_complex)

    # Assessment
    logger.info("Generating quality assessment...")
    assessment = generate_quality_assessment(stats)

    elapsed_time = time.time() - start_time
    stats["analysis_duration_s"] = elapsed_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    logger.info(f"Quality score: {assessment['quality_score']:.1f}/100")

    return {**stats, "quality_assessment": assessment}


def main():
    parser = argparse.ArgumentParser(
        description="I/Q Data Sanity Checker for RF Signal Processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("wav", type=str, help="Input WAV file containing I/Q data")
    parser.add_argument("--fs_hint", type=float, default=None, help="Override sample rate from WAV header (Hz)")
    parser.add_argument("--out_dir", type=str, default="out_report", help="Base output directory for results")
    parser.add_argument("--fft_size", type=int, default=None,
                        help=f"FFT size for IRR analysis (default: adaptive, max {MAX_FFT_SIZE})")
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help="Maximum samples to load for large files (None for all)")
    parser.add_argument("--windows", type=int, default=16,
                        help="Number of evenly spaced windows to analyze (0 to disable)")
    parser.add_argument("--window_size", type=int, default=DEFAULT_FFT_SIZE,
                        help="Samples per window (per channel) for windowed analysis")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output (errors only)")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    try:
        input_file = validate_input_file(args.wav)
        run_dir = create_unique_run_directory(args.out_dir, input_file)

        if args.fft_size is not None:
            if args.fft_size < MIN_FFT_SIZE or args.fft_size > MAX_FFT_SIZE:
                raise SanityCheckError(f"FFT size must be between {MIN_FFT_SIZE} and {MAX_FFT_SIZE}")
            if args.fft_size & (args.fft_size - 1) != 0:
                raise SanityCheckError("FFT size must be a power of 2")

        results = perform_sanity_check(
            wav_file=input_file,
            fs_hint=args.fs_hint,
            fft_size=args.fft_size,
            max_samples=args.max_samples if (args.windows is None or args.windows <= 0) else None,
            output_dir=run_dir,
            n_windows=args.windows,
            window_size=args.window_size
        )

        stats_file, report_file = generate_output_files(
            stats=results,
            assessment=results["quality_assessment"],
            input_file=input_file,
            output_dir=run_dir
        )

        # Save run metadata
        try:
            meta = {
                "run_dir": str(run_dir),
                "input_file": str(input_file),
                "args": {
                    "fs_hint": args.fs_hint,
                    "out_dir_base": args.out_dir,
                    "fft_size": args.fft_size,
                    "max_samples": args.max_samples,
                    "windows": args.windows,
                    "window_size": args.window_size,
                    "verbose": args.verbose,
                    "quiet": args.quiet,
                },
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to write run metadata: {e}")

        if not args.quiet:
            print(json.dumps(results, indent=2))
            print(f'\n[INFO] Outputs written to: "{run_dir}"')

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
        # Optional traceback in verbose mode
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

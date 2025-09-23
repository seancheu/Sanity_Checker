#!/usr/bin/env python3
"""
I/Q Data Sanity Checker for RF Signal Processing Pipeline (streaming full-file)

Key features:
- Per-run unique output subfolder under --out_dir
- UTF-8-safe outputs
- Streams entire file in bounded RAM using a computed chunk size from --mem_budget_mb
- Pass 1: Full-file stats + IRR for all I/Q conventions (accumulate pos/neg power)
- Pass 2: Full-file envelope kurtosis under best convention via online moments

Example:
    python 1_sanity_check_iq_v4.py your.wav --fs_hint 20000000 --mem_budget_mb 512
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

# Configuration constants
DEFAULT_FFT_SIZE: int = 1 << 18   # 262144
MIN_FFT_SIZE: int = 1 << 12       # 4096
MAX_FFT_SIZE: int = 1 << 20       # 1048576
CLIP_THRESHOLD: float = 0.999
EPSILON: float = 1e-12
DEFAULT_IRR_THRESHOLD: float = 20.0

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SanityCheckError(Exception):
    pass

# --------- Path helpers ---------

def validate_input_file(file_path: Union[str, Path]) -> Path:
    p = Path(file_path)
    if not p.exists():
        raise SanityCheckError(f"Input file does not exist: {p}")
    if not p.is_file():
        raise SanityCheckError(f"Input path is not a file: {p}")
    if not os.access(p, os.R_OK):
        raise SanityCheckError(f"Input file is not readable: {p}")
    return p

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

# --------- Streaming math helpers ---------

def calculate_adaptive_fft_size(num_samples: int, target_size: int = DEFAULT_FFT_SIZE) -> int:
    max_size = min(target_size, num_samples)
    if max_size < 2:
        return MIN_FFT_SIZE
    fft_size = 1 << (max_size.bit_length() - 1)
    return max(MIN_FFT_SIZE, min(MAX_FFT_SIZE, fft_size))

class RunningStats:
    """
    Online mean/std/min/max/rms & clip counting for a single channel.
    Uses Welford for mean/std; RMS via sum of squares.
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.minv = float('inf')
        self.maxv = float('-inf')
        self.sum_sq = 0.0
        self.clip_count = 0
        self.nan_count = 0
        self.inf_count = 0

    def update(self, x: np.ndarray):
        # Count nans/infs and sanitize for stats
        n_nan = np.isnan(x).sum()
        n_inf = np.isinf(x).sum()
        self.nan_count += int(n_nan)
        self.inf_count += int(n_inf)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        # Clip count
        self.clip_count += int(np.sum(np.abs(x) > CLIP_THRESHOLD))
        # Min/Max
        self.minv = float(min(self.minv, np.min(x)))
        self.maxv = float(max(self.maxv, np.max(x)))
        # Sum of squares (for RMS)
        self.sum_sq += float(np.sum(x.astype(np.float64)**2))
        # Welford updates
        n_old = self.n
        self.n += x.size
        delta = float(np.mean(x)) - self.mean
        self.mean += (x.size * delta) / self.n
        # For M2 with batch: sum((xi - new_mean)*(xi - old_mean))
        # We can compute M2 incrementally using per-batch variance:
        var_batch = float(np.var(x, dtype=np.float64))
        self.M2 += var_batch * (x.size - 1) + (n_old * x.size / self.n) * (delta ** 2)

    def finalize(self, label: str) -> Dict[str, float]:
        std = float(np.sqrt(self.M2 / self.n)) if self.n > 1 else 0.0
        rms = float(np.sqrt(self.sum_sq / self.n)) if self.n > 0 else 0.0
        clip_pct = 100.0 * self.clip_count / self.n if self.n > 0 else 0.0
        return {
            f"{label}_mean": float(self.mean),
            f"{label}_std": std,
            f"{label}_peak_abs": float(max(abs(self.minv), abs(self.maxv))) if self.n > 0 else 0.0,
            f"{label}_clip_pct": clip_pct,
            f"{label}_min": float(self.minv if self.n > 0 else 0.0),
            f"{label}_max": float(self.maxv if self.n > 0 else 0.0),
            f"{label}_rms": rms,
            "nan_count": self.nan_count,   # aggregated outside for both channels
            "inf_count": self.inf_count,
        }

class OnlineKurtosis:
    """
    Online kurtosis via running raw moments (stable enough for large N).
    Computes population kurtosis E[((X-μ)/σ)^4].
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.M3 = 0.0
        self.M4 = 0.0

    def update(self, x: np.ndarray):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        x = x.astype(np.float64, copy=False)
        n1 = self.n
        n = n1 + x.size
        delta = np.mean(x) - self.mean
        delta2 = delta * delta
        delta3 = delta2 * delta
        delta4 = delta2 * delta2
        m2_batch = float(np.var(x, ddof=0))
        m3_batch = float(((x - np.mean(x))**3).mean())
        m4_batch = float(((x - np.mean(x))**4).mean())

        # Pébay (2008)-style pooled moments for combining two sets
        if n1 == 0:
            self.mean = float(np.mean(x))
            self.M2 = m2_batch * x.size
            self.M3 = m3_batch * x.size
            self.M4 = m4_batch * x.size
            self.n = x.size
            return

        mean_new = self.mean + (x.size * delta) / n

        M2 = self.M2 + x.size * m2_batch + (n1 * x.size / n) * delta2
        M3 = (self.M3 + x.size * m3_batch
              + (n1 * x.size / n) * delta3
              + 3.0 * (self.M2 * x.size / n) * delta
              - 3.0 * (x.size * m2_batch * n1 / n) * delta)
        M4 = (self.M4 + x.size * m4_batch
              + (n1 * x.size / n) * delta4
              + 6.0 * (self.M2 * x.size / n) * delta2
              + 4.0 * (self.M3 * x.size / n) * delta
              - 4.0 * (x.size * m3_batch * n1 / n) * delta
              - 6.0 * (x.size * m2_batch * n1 / n) * delta2)

        self.mean = mean_new
        self.M2 = M2
        self.M3 = M3
        self.M4 = M4
        self.n = n

    def kurtosis(self) -> float:
        if self.n < 2 or self.M2 <= 0:
            return 0.0
        var = self.M2 / self.n
        if var <= 0:
            return 0.0
        return float((self.M4 / self.n) / (var ** 2))

# --------- IRR helpers ---------

def irr_pos_neg_powers(complex_signal: np.ndarray, fft_size: int) -> Tuple[float, float]:
    """Return (pos_power, neg_power) from a segment."""
    if complex_signal.size < fft_size:
        # center-pad with zeros if needed
        pad = fft_size - complex_signal.size
        left = pad // 2
        right = pad - left
        complex_signal = np.pad(complex_signal, (left, right), mode='constant')
    else:
        # center crop
        start = (complex_signal.size - fft_size) // 2
        complex_signal = complex_signal[start:start+fft_size]

    X = np.fft.fftshift(np.fft.fft(complex_signal, n=fft_size))
    P = (np.abs(X) ** 2).astype(np.float64)
    mid = P.size // 2
    pos = float(P[mid+1:].sum() + EPSILON)
    neg = float(P[:mid].sum() + EPSILON)
    return pos, neg

# --------- Streaming passes ---------

def compute_chunk_size_samples(mem_budget_mb: int, channels: int, floor_fft: int) -> int:
    """
    Compute a safe chunk size (frames) so that chunk * channels * 4 bytes <= mem_budget_mb,
    and also ensure it's >= floor_fft for FFT analysis. Round up to a power-of-two-ish size.
    """
    mem_bytes = max(mem_budget_mb, 64) * 1024 * 1024  # at least 64 MB
    max_frames = mem_bytes // (channels * 4)          # float32
    if max_frames < floor_fft:
        # If budget is tiny, fall back to floor_fft
        chunk = floor_fft
    else:
        chunk = int(max_frames)
        # snap to multiple of floor_fft to avoid too many fractional segments
        k = max(1, chunk // floor_fft)
        chunk = k * floor_fft
    # keep a sane upper bound to avoid extremely long per-FFT time
    return int(min(chunk, 16 * 1024 * 1024))  # cap ~16M frames per chunk

def pass1_stats_and_irr(filepath: Path, fs: float, fft_size: int, chunk_frames: int) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Stream over the entire file:
    - Running stats for I and Q (mean/std/min/max/rms/clip)
    - Accumulate pos/neg powers for each convention for IRR
    Returns:
      stats_partial (with per-channel stats & counters),
      irr_totals: dict of total pos/neg per convention
    """
    info = sf.info(filepath)
    total_frames = int(info.frames)
    channels = info.channels
    if channels < 2:
        raise SanityCheckError(f"Need 2 channels (I,Q), got {channels}")

    I_stats = RunningStats()
    Q_stats = RunningStats()

    # Accumulators for IRR (per convention)
    irr_acc = {
        "I_Q": {"pos": 0.0, "neg": 0.0},
        "I_neg_Q": {"pos": 0.0, "neg": 0.0},
        "Q_I": {"pos": 0.0, "neg": 0.0},
    }

    # Iterate chunks
    start = 0
    logger.info(f"Pass 1: streaming stats + IRR (fft_size={fft_size}, chunk_frames={chunk_frames})")
    while start < total_frames:
        stop = min(start + chunk_frames, total_frames)
        data = sf.read(filepath, start=start, stop=stop, always_2d=True, dtype='float32')
        if isinstance(data, tuple):
            data = data[0]

        # Update channel stats (sanitize NaN/Inf inside RunningStats)
        I = data[:, 0]
        Q = data[:, 1]
        I_stats.update(I)
        Q_stats.update(Q)

        # For IRR, remove DC per chunk for stability
        I_dc = I - float(np.mean(I[np.isfinite(I)])) if np.isfinite(I).any() else I
        Q_dc = Q - float(np.mean(Q[np.isfinite(Q)])) if np.isfinite(Q).any() else Q

        # Conventions
        sig_I_Q     = I_dc + 1j * Q_dc
        sig_I_neg_Q = I_dc - 1j * Q_dc
        sig_Q_I     = Q_dc + 1j * I_dc

        for key, sig_c in (("I_Q", sig_I_Q), ("I_neg_Q", sig_I_neg_Q), ("Q_I", sig_Q_I)):
            pos, neg = irr_pos_neg_powers(sig_c, fft_size)
            irr_acc[key]["pos"] += pos
            irr_acc[key]["neg"] += neg

        start = stop

    # Extract stats
    I_final = I_stats.finalize("I")
    Q_final = Q_stats.finalize("Q")

    # combine NaN/Inf counts
    nan_total = I_final["nan_count"] + Q_final["nan_count"]
    inf_total = I_final["inf_count"] + Q_final["inf_count"]
    del I_final["nan_count"]; del I_final["inf_count"]
    del Q_final["nan_count"]; del Q_final["inf_count"]

    stats_partial = {
        **I_final,
        **Q_final,
        "nan_count": nan_total,
        "inf_count": inf_total,
    }

    return stats_partial, {
        "irr_db_I_Q": 10.0 * np.log10((irr_acc["I_Q"]["pos"]) / (irr_acc["I_Q"]["neg"] + EPSILON)),
        "irr_db_I_neg_Q": 10.0 * np.log10((irr_acc["I_neg_Q"]["pos"]) / (irr_acc["I_neg_Q"]["neg"] + EPSILON)),
        "irr_db_Q_I": 10.0 * np.log10((irr_acc["Q_I"]["pos"]) / (irr_acc["Q_I"]["neg"] + EPSILON)),
    }

def pass2_envelope_kurtosis(filepath: Path, fs: float, best_convention: str, chunk_frames: int) -> float:
    """
    Stream again to compute envelope kurtosis under the chosen convention.
    """
    info = sf.info(filepath)
    total_frames = int(info.frames)
    channels = info.channels
    if channels < 2:
        raise SanityCheckError(f"Need 2 channels (I,Q), got {channels}")

    ok = OnlineKurtosis()
    start = 0
    logger.info(f"Pass 2: envelope kurtosis (best convention: {best_convention}, chunk_frames={chunk_frames})")
    while start < total_frames:
        stop = min(start + chunk_frames, total_frames)
        data = sf.read(filepath, start=start, stop=stop, always_2d=True, dtype='float32')
        if isinstance(data, tuple):
            data = data[0]
        I = data[:, 0]
        Q = data[:, 1]

        # DC removal per chunk to stabilize envelope
        I_dc = I - float(np.mean(I[np.isfinite(I)])) if np.isfinite(I).any() else I
        Q_dc = Q - float(np.mean(Q[np.isfinite(Q)])) if np.isfinite(Q).any() else Q

        if best_convention == "I+jQ":
            z = I_dc + 1j * Q_dc
        elif best_convention == "I-jQ":
            z = I_dc - 1j * Q_dc
        else:  # "Q+jI"
            z = Q_dc + 1j * I_dc

        env = np.abs(z).astype(np.float64, copy=False)
        ok.update(env)

        start = stop

    return ok.kurtosis()

# --------- Assessment & outputs ---------

def generate_quality_assessment(stats: Dict[str, Any]) -> Dict[str, Any]:
    assessment = {"quality_score": 100.0, "warnings": [], "recommendations": [], "issues": []}

    if stats.get("nan_count", 0) > 0:
        assessment["issues"].append(f"Found {stats['nan_count']} NaN values")
        assessment["quality_score"] -= 20
    if stats.get("inf_count", 0) > 0:
        assessment["issues"].append(f"Found {stats['inf_count']} infinite values")
        assessment["quality_score"] -= 20

    max_clip = max(stats.get("I_clip_pct", 0.0), stats.get("Q_clip_pct", 0.0))
    if max_clip > 5.0:
        assessment["issues"].append(f"Severe clipping detected: {max_clip:.1f}%")
        assessment["quality_score"] -= 25
    elif max_clip > 1.0:
        assessment["warnings"].append(f"Moderate clipping detected: {max_clip:.1f}%")
        assessment["quality_score"] -= 10
    elif max_clip > 0.1:
        assessment["warnings"].append(f"Minor clipping detected: {max_clip:.1f}%")
        assessment["quality_score"] -= 5

    best_irr = stats.get("best_irr_db", 0.0)
    if best_irr < 10:
        assessment["issues"].append(f"Poor I/Q balance (IRR: {best_irr:.1f} dB)")
        assessment["quality_score"] -= 15
    elif best_irr < DEFAULT_IRR_THRESHOLD:
        assessment["warnings"].append(f"Suboptimal I/Q balance (IRR: {best_irr:.1f} dB)")
        assessment["quality_score"] -= 5

    if stats.get("best_convention", "I+jQ") != "I+jQ":
        bc = stats["best_convention"]
        assessment["recommendations"].append(
            f"Consider using --conv {bc.replace('+j', ' ').replace('-j', ' -').replace('j', '')} in downstream processing"
        )

    # Full-file analysis → covered_duration == file_duration, so no "very short" warning
    kurt = stats.get("envelope_kurtosis", 3.0)
    if kurt < 1.5:
        assessment["warnings"].append("Very constant envelope detected - may indicate CW or unmodulated carrier")
    elif kurt > 10:
        assessment["warnings"].append("Highly variable envelope detected - check for intermittent signals or interference")

    assessment["quality_score"] = max(0.0, assessment["quality_score"])
    return assessment

def generate_output_files(stats: Dict[str, Any], assessment: Dict[str, Any],
                          input_file: Path, output_dir: Path) -> Tuple[Path, Path]:
    base_name = input_file.stem
    stats_file = output_dir / f"sanity_check_{base_name}_stats.json"
    report_file = output_dir / f"sanity_check_{base_name}_report.txt"

    # JSON
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({**stats, "quality_assessment": assessment}, f, indent=2, sort_keys=True, ensure_ascii=False)
    logger.info(f"Statistics written to: {stats_file}")

    # TXT report
    with open(report_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("I/Q Data Sanity Check Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input File: {input_file}\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Quality Score: {assessment['quality_score']:.1f}/100\n\n")

        f.write("File Characteristics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"File duration: {stats['file_duration_s']:.3f} seconds\n")
        f.write(f"Covered duration: {stats['covered_duration_s']:.3f} seconds\n")
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
        f.write(f"I channel clipping: {stats.get('I_clip_pct', 0.0):.2f}%\n")
        f.write(f"Q channel clipping: {stats.get('Q_clip_pct', 0.0):.2f}%\n\n")

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
    return stats_file, report_file

# --------- Orchestration ---------

def perform_sanity_check_streaming(wav_file: Path, fs_hint: Optional[float],
                                   fft_size: Optional[int],
                                   mem_budget_mb: int,
                                   output_dir: Path) -> Dict[str, Any]:
    logger.info(f"Starting sanity check (streaming) for: {wav_file}")

    # Probe file
    info = sf.info(wav_file)
    total_frames = int(info.frames)
    channels = info.channels
    sr_header = float(info.samplerate)
    fs = fs_hint if fs_hint is not None else sr_header
    if channels < 2:
        raise SanityCheckError(f"Need 2 channels (I,Q), got {channels}")

    file_duration_s = total_frames / fs

    # Determine per-chunk frames from memory budget
    floor_fft = calculate_adaptive_fft_size(fft_size if fft_size else DEFAULT_FFT_SIZE)
    chunk_frames = compute_chunk_size_samples(mem_budget_mb, channels=channels, floor_fft=floor_fft)
    use_fft = calculate_adaptive_fft_size(chunk_frames if not fft_size else fft_size)

    logger.info(f"Detected: frames={total_frames:,}, duration={file_duration_s:.2f}s, channels={channels}, fs={fs:,.0f} Hz")
    logger.info(f"Chunk planning: mem_budget_mb={mem_budget_mb}, chunk_frames={chunk_frames:,}, fft_size={use_fft}")

    start_time = time.time()

    # Pass 1: stats + IRR totals across full file
    ch_stats_partial, irr_db_map = pass1_stats_and_irr(wav_file, fs, use_fft, chunk_frames)

    # Decide best convention
    best_key = max(irr_db_map, key=lambda k: irr_db_map[k])
    if best_key == "I_Q":
        best_convention = "I+jQ"
    elif best_key == "I_neg_Q":
        best_convention = "I-jQ"
    else:
        best_convention = "Q+jI"
    best_irr_db = float(abs(irr_db_map[best_key]))

    # Pass 2: envelope kurtosis under best convention (full file)
    env_kurt = pass2_envelope_kurtosis(wav_file, fs, best_convention, chunk_frames)

    elapsed = time.time() - start_time

    # Build stats
    I_only = {k: v for k, v in ch_stats_partial.items() if k.startswith("I_")}
    Q_only = {k: v for k, v in ch_stats_partial.items() if k.startswith("Q_")}
    nan_total = ch_stats_partial["nan_count"]
    inf_total = ch_stats_partial["inf_count"]

    stats = {
        "analysis_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "input_file": str(wav_file),
        "frames": int(total_frames),
        "channels": int(channels),
        "fs_header_hz": float(sr_header),
        "fs_used_hz": float(fs),
        "file_duration_s": float(file_duration_s),
        "covered_duration_s": float(file_duration_s),  # full coverage
        "coverage_fraction": 1.0,
        "nan_count": int(nan_total),
        "inf_count": int(inf_total),
        "sampling_strategy": f"stream_full(mem_budget_mb={mem_budget_mb}, chunk_frames={chunk_frames})",
        "analysis_params": {
            "fft_size": int(use_fft),
            "mem_budget_mb": int(mem_budget_mb),
            "chunk_frames": int(chunk_frames),
            "passes": 2
        },
        **I_only,
        **Q_only,
        "best_convention": best_convention,
        "best_irr_db": best_irr_db,
        "envelope_kurtosis": float(env_kurt),
        "analysis_duration_s": float(elapsed)
    }

    assessment = generate_quality_assessment(stats)
    logger.info(f"Analysis completed in {elapsed:.2f} seconds")
    logger.info(f"Quality score: {assessment['quality_score']:.1f}/100")

    return {**stats, "quality_assessment": assessment}

# --------- CLI ---------

def main():
    parser = argparse.ArgumentParser(
        description="I/Q Data Sanity Checker (streaming full-file)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("wav", type=str, help="Input WAV file containing I/Q data")
    parser.add_argument("--fs_hint", type=float, default=None, help="Override sample rate from WAV header (Hz)")
    parser.add_argument("--out_dir", type=str, default="out_report", help="Base output directory for results")
    parser.add_argument("--fft_size", type=int, default=None,
                        help=f"FFT size for IRR analysis per chunk (default adaptive, max {MAX_FFT_SIZE})")
    parser.add_argument("--mem_budget_mb", type=int, default=512,
                        help="Approx RAM budget for one chunk (MB). Larger = fewer, bigger chunks.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Errors only")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    try:
        input_file = validate_input_file(args.wav)

        # Ensure fft_size sanity if provided
        if args.fft_size is not None:
            if args.fft_size < MIN_FFT_SIZE or args.fft_size > MAX_FFT_SIZE or (args.fft_size & (args.fft_size - 1)):
                raise SanityCheckError(f"fft_size must be a power of 2 within [{MIN_FFT_SIZE}, {MAX_FFT_SIZE}]")

        # Unique per-run folder
        run_dir = create_unique_run_directory(args.out_dir, input_file)

        # Run streaming analysis
        results = perform_sanity_check_streaming(
            wav_file=input_file,
            fs_hint=args.fs_hint,
            fft_size=args.fft_size,
            mem_budget_mb=max(64, args.mem_budget_mb),
            output_dir=run_dir
        )

        # Write outputs
        stats_file, report_file = generate_output_files(
            stats=results,
            assessment=results["quality_assessment"],
            input_file=input_file,
            output_dir=run_dir
        )

        # Metadata
        try:
            meta = {
                "run_dir": str(run_dir),
                "input_file": str(input_file),
                "args": {
                    "fs_hint": args.fs_hint,
                    "out_dir_base": args.out_dir,
                    "fft_size": args.fft_size,
                    "mem_budget_mb": args.mem_budget_mb,
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

        # Final status
        quality = results["quality_assessment"]["quality_score"]
        if quality >= 80:
            logger.info("[OK] I/Q data quality is good")
        elif quality >= 60:
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
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

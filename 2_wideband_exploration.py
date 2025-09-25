#python analyze_iq_unknown.py your_20MHz.wav --fs_hint 20000000 --out out_report --nperseg 4096 --overlap 0.5 --prom_db 8 --cfar_k 3.0

#If the PSD marks too many tiny peaks, raise --prom_db to 10-12.
#If the waterfall mask is too "speckly", raise --cfar_k to 3.5-4.0 (stricter).
#If Step-1 told you to swap/flip I/Q, add --conv I-Q or --conv Q+I.

# analyze_iq_unknown.py - Step 2: PSD, waterfall, occupancy, carrier table
import argparse, json, csv, os, glob
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal as sig
import matplotlib.pyplot as plt
import time
import sys
import math

# ---------- IRR Calculation (from Step 1) ----------
EPSILON = 1e-12
DEFAULT_FFT_SIZE = 1 << 18  # 262144

def calculate_adaptive_fft_size(num_samples: int, target_size: int = DEFAULT_FFT_SIZE) -> int:
    max_size = min(target_size, num_samples)
    if max_size < 2:
        return 1 << 12  # MIN_FFT_SIZE = 4096
    fft_size = 1 << (max_size.bit_length() - 1)
    return max(1 << 12, min(1 << 20, fft_size))  # MIN=4096, MAX=1048576

def irr_pos_neg_powers(complex_signal: np.ndarray, fft_size: int):
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

def calculate_irr_db(xc: np.ndarray, fft_size: int = None) -> float:
    """Calculate IRR in dB for complex I/Q signal."""
    if fft_size is None:
        fft_size = calculate_adaptive_fft_size(len(xc))

    # Remove DC for stability
    xc_dc = xc - np.mean(xc)

    # Calculate pos/neg powers
    pos, neg = irr_pos_neg_powers(xc_dc, fft_size)

    # Return IRR in dB
    return 10.0 * np.log10(pos / (neg + EPSILON))

def read_step1_irr(step1_output_dir: str) -> dict:
    """Read IRR values from Step 1 output if available."""
    try:
        # Look for Step 1 JSON files
        pattern = os.path.join(step1_output_dir, "**/sanity_check_*_stats.json")
        json_files = glob.glob(pattern, recursive=True)

        if not json_files:
            print_progress("No Step 1 IRR data found")
            return {}

        # Use the most recent file
        latest_file = max(json_files, key=os.path.getctime)

        with open(latest_file, 'r') as f:
            step1_data = json.load(f)

        return {
            "step1_best_convention": step1_data.get("best_convention", "Unknown"),
            "step1_best_irr_db": step1_data.get("best_irr_db", 0.0),
            "step1_file": latest_file
        }
    except Exception as e:
        print_progress(f"Could not read Step 1 IRR data: {e}")
        return {}

# ---------- Utils ----------
def block_reduce_2d(A, r_f: int = 1, r_t: int = 1, op: str = "max"):
    """
    Downsample 2D array A[F, T] by integer factors r_f, r_t using block reduction.
    Pads the array so it divides evenly, then applies np.nanmax or np.nanmean.
    Returns (A_ds, f_idx, t_idx) where idx arrays map reduced bins to original indices.
    """
    F, T = A.shape
    r_f = max(1, int(r_f))
    r_t = max(1, int(r_t))

    pad_f = (-F) % r_f
    pad_t = (-T) % r_t
    if pad_f or pad_t:
        A = np.pad(A, ((0, pad_f), (0, pad_t)), mode="edge")
        Fp, Tp = A.shape
    else:
        Fp, Tp = F, T

    A = A.reshape(Fp // r_f, r_f, Tp // r_t, r_t)
    if op == "mean":
        A_ds = np.nanmean(np.nanmean(A, axis=3), axis=1)
    else:  # "max" is good for preserving bursts
        A_ds = np.nanmax(np.nanmax(A, axis=3), axis=1)

    # Build index mapping for axes (take centers of blocks)
    f_idx = (np.arange(A_ds.shape[0]) * r_f + min(r_f // 2, r_f - 1)).clip(0, F - 1)
    t_idx = (np.arange(A_ds.shape[1]) * r_t + min(r_t // 2, r_t - 1)).clip(0, T - 1)
    return A_ds, f_idx.astype(int), t_idx.astype(int)


def make_waterfall_thumbnail(S_db, f, t, max_cols: int = 4000, max_rows: int = None, op: str = "max"):
    """
    Produce a memory-friendly thumbnail of waterfall.
    - Limit time columns to max_cols (downsample along T).
    - Optionally limit frequency rows to max_rows.
    Returns (S_db_vis, f_vis, t_vis)
    """
    F, T = S_db.shape
    r_t = max(1, math.ceil(T / max_cols)) if max_cols else 1
    r_f = 1
    if max_rows:
        r_f = max(1, math.ceil(F / max_rows))

    S_small, f_idx, t_idx = block_reduce_2d(S_db, r_f=r_f, r_t=r_t, op=op)
    # Map indices to actual axis values
    f_vis = f[f_idx] if len(f) == F else np.linspace(f.min(), f.max(), S_small.shape[0])
    t_vis = t[t_idx] if len(t) == T else np.linspace(t.min(), t.max(), S_small.shape[1])
    # Keep dtype light
    return S_small.astype(np.float32, copy=False), f_vis.astype(np.float32), t_vis.astype(np.float32)

def print_progress(msg, elapsed=None):
    """Print progress with optional timing info"""
    if elapsed is not None:
        print(f"[{elapsed:.1f}s] {msg}", flush=True)
    else:
        print(f"[INFO] {msg}", flush=True)

def estimate_memory_usage(N, nperseg, overlap):
    """Estimate memory usage for processing"""
    noverlap = int(nperseg * overlap)
    hop = nperseg - noverlap
    n_windows = (N - noverlap) // hop

    # Memory for STFT: F x T complex values (8 bytes each)
    stft_memory = nperseg * n_windows * 8

    # Memory for waterfall dB: F x T float32 (4 bytes each)
    waterfall_memory = nperseg * n_windows * 4

    # Additional memory overhead for processing
    overhead_memory = stft_memory * 0.5  # 50% overhead for intermediate calculations

    total_mb = (stft_memory + waterfall_memory + overhead_memory) / (1024 * 1024)
    return total_mb, n_windows

def check_memory_safety(estimated_mb, max_safe_mb=2048):
    """Check if estimated memory usage is safe"""
    if estimated_mb > max_safe_mb:
        raise RuntimeError(f"Estimated memory usage {estimated_mb:.1f} MB exceeds safe limit {max_safe_mb} MB. "
                         f"Consider using --max_duration to reduce file size or --force_full to override (dangerous).")
    elif estimated_mb > max_safe_mb * 0.5:
        print_progress(f"WARNING: High memory usage estimated: {estimated_mb:.1f} MB")

def test_irr_sampling_factor(I, Q, factor, best_convention, max_test_samples=1_000_000):
    """Test IRR quality for a specific sampling factor."""
    # Use a subset for quick testing
    test_samples = min(len(I), max_test_samples)
    I_test = I[:test_samples:factor]
    Q_test = Q[:test_samples:factor]

    if len(I_test) < 1000:  # Need minimum samples for reliable IRR
        return float('-inf')

    try:
        if best_convention == "I+jQ":
            xc = (I_test - np.mean(I_test)) + 1j * (Q_test - np.mean(Q_test))
        elif best_convention == "I-jQ":
            xc = (I_test - np.mean(I_test)) - 1j * (Q_test - np.mean(Q_test))
        elif best_convention == "Q+jI":
            xc = (Q_test - np.mean(Q_test)) + 1j * (I_test - np.mean(I_test))
        else:  # Q-jI
            xc = (Q_test - np.mean(Q_test)) - 1j * (I_test - np.mean(I_test))

        return calculate_irr_db(xc)
    except:
        return float('-inf')

def adaptive_sampling_strategy(N, fs, I, Q, best_convention, target_duration=60.0, min_samples=10e6):
    """
    Adaptive sampling strategy that preserves IRR quality.
    Tests multiple sampling factors and chooses the best IRR result.
    Returns (use_sampling, sample_indices, effective_fs, reduction_factor, chosen_factor)
    """
    duration = N / fs

    # If file is small enough, process entirely
    if duration <= target_duration or N <= min_samples:
        return False, None, fs, 1.0, 1

    # Calculate target reduction factor
    target_reduction = duration / target_duration

    # Generate candidate sampling factors with preference for IRR-friendly ones
    candidates = []

    # Start with factors around the target, prioritizing odd numbers and powers of 2 >= 8
    base_factor = int(np.ceil(target_reduction))

    # Test range around target factor
    test_range = max(2, int(base_factor * 0.5)), min(20, int(base_factor * 2))

    for factor in range(test_range[0], test_range[1] + 1):
        # Prioritize IRR-friendly factors
        priority = 0
        if factor % 2 == 1:  # Odd numbers (good for IRR)
            priority += 10
        elif factor >= 8 and (factor & (factor - 1)) == 0:  # Powers of 2 >= 8 (good)
            priority += 5
        elif factor in [3, 5, 7, 9, 15]:  # Known good factors
            priority += 8
        # Even factors 2, 4, 6, 12, 14, 16 get lower priority (bad for IRR)

        candidates.append((factor, priority))

    # Sort by priority (higher is better), then by closeness to target
    candidates.sort(key=lambda x: (-x[1], abs(x[0] - base_factor)))

    print_progress(f"Large file detected: {duration:.1f}s ({N/1e6:.1f}M samples)")
    print_progress(f"Testing sampling factors for optimal IRR...")

    best_factor = base_factor
    best_irr = float('-inf')
    irr_results = {}

    # Test top candidates
    for factor, priority in candidates[:8]:  # Test up to 8 factors
        irr_db = test_irr_sampling_factor(I, Q, factor, best_convention)
        irr_results[factor] = irr_db

        if irr_db > best_irr:
            best_irr = irr_db
            best_factor = factor

        # Show results for user feedback
        status = "GOOD" if irr_db > 10 else "OK" if irr_db > 0 else "POOR"
        effective_fs_mhz = fs / factor / 1e6
        print_progress(f"  1:{factor:2d} sampling ({effective_fs_mhz:5.1f} MHz): {irr_db:+6.1f} dB [{status}]")

    # Generate sampling indices
    sample_indices = np.arange(0, N, best_factor)
    effective_fs = fs / best_factor
    reduction_factor = best_factor

    print_progress(f"Selected: 1:{best_factor} sampling (IRR: {best_irr:+.1f} dB) -> {len(sample_indices)/1e6:.1f}M samples")

    return True, sample_indices, effective_fs, reduction_factor, best_factor

def smart_sampling_strategy(N, fs, target_duration=60.0, min_samples=10e6):
    """
    Determine smart sampling strategy for large files.
    Returns (use_sampling, sample_indices, effective_fs, reduction_factor)
    """
    duration = N / fs

    # If file is small enough, process entirely
    if duration <= target_duration or N <= min_samples:
        return False, None, fs, 1.0

    # Calculate reduction factor to hit target duration
    reduction_factor = duration / target_duration

    # Use systematic sampling to preserve spectral characteristics
    step = int(np.ceil(reduction_factor))
    sample_indices = np.arange(0, N, step)
    effective_fs = fs / step

    print_progress(f"Large file detected: {duration:.1f}s ({N/1e6:.1f}M samples)")
    print_progress(f"Using 1:{step} sampling -> {len(sample_indices)/1e6:.1f}M samples, {target_duration:.1f}s effective")

    return True, sample_indices, effective_fs, reduction_factor

def to_complex_iq(x, conv="I+Q"):
    """
    x: float32 [N, C>=2], conv in {"I+Q","I-Q","Q+I"} from Step-1 result.
    """
    I, Q = x[:,0].astype(np.float32), x[:,1].astype(np.float32)
    if conv == "I+Q":   xc = I + 1j*Q
    elif conv == "I-Q": xc = I - 1j*Q
    elif conv == "Q+I": xc = Q + 1j*I
    else: raise ValueError(f"Unknown conv '{conv}'")
    return xc

def dc_remove(xc):
    return xc - np.mean(xc)

def read_with_chunked_sampling(file_path, reduction_factor):
    """
    Read WAV file with chunked sampling for memory efficiency.
    Applies systematic sampling (every Nth sample) while reading in chunks.
    """
    # Get file info without loading
    info = sf.info(file_path)
    total_frames = info.frames
    channels = info.channels

    # Calculate sampling parameters
    step = int(np.ceil(reduction_factor))
    expected_samples = total_frames // step

    # Determine chunk size (1M samples or 10% of file, whichever is smaller)
    chunk_size = min(1024 * 1024, total_frames // 10)

    print_progress(f"Chunked sampling: step={step}, chunk_size={chunk_size:,}, expected_output={expected_samples/1e6:.1f}M samples")

    # Pre-allocate output array
    x_sampled = np.zeros((expected_samples, channels), dtype=np.float32)

    samples_read = 0
    samples_written = 0

    with sf.SoundFile(file_path, 'r') as sfile:
        while samples_read < total_frames and samples_written < expected_samples:
            # Calculate chunk boundaries
            chunk_start = samples_read
            chunk_end = min(samples_read + chunk_size, total_frames)
            actual_chunk_size = chunk_end - chunk_start

            # Read chunk
            chunk_data = sfile.read(actual_chunk_size, always_2d=True)

            # Apply systematic sampling to chunk
            # Start sampling from the correct offset within the chunk
            start_offset = (step - (samples_read % step)) % step
            subsampled_chunk = chunk_data[start_offset::step]

            # Write to output array
            end_idx = min(samples_written + len(subsampled_chunk), expected_samples)
            actual_write_size = end_idx - samples_written
            x_sampled[samples_written:end_idx] = subsampled_chunk[:actual_write_size]

            samples_written += actual_write_size
            samples_read = chunk_end

            # Log progress for large files
            if chunk_end % (chunk_size * 5) == 0:
                progress_pct = 100 * samples_read / total_frames
                print_progress(f"Reading progress: {progress_pct:.1f}% ({samples_read:,}/{total_frames:,} frames)")

    print_progress(f"Chunked sampling complete: {samples_written:,} samples written")
    return x_sampled[:samples_written]

def welch_psd(xc, fs, nperseg, noverlap):
    f, Pxx = sig.welch(
        xc, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
        return_onesided=False, detrend=False, scaling="density"
    )
    f = np.fft.fftshift(f); Pxx = np.fft.fftshift(Pxx)
    return f, Pxx

def stft_waterfall(xc, fs, nperseg, noverlap, max_cols=4000, window="hann"):
    """
    Memory-safe STFT:
      - Computes STFT frames in a streaming loop (no giant 2D arrays).
      - Skips frames so total time columns <= max_cols.
      - Uses complex64/float32 to keep memory light.
      - Memory monitoring and chunked allocation to prevent crashes.
    Returns:
      f [F], t_vis [T], S_db [F,T]  (already 'fftshift'-ed, ready to plot)
    """

    # Ensure light dtypes
    x = np.asarray(xc, dtype=np.complex64, order="C")
    N = x.size

    win = sig.get_window(window, nperseg, fftbins=True).astype(np.float32)
    hop = nperseg - int(nperseg * noverlap)
    if hop <= 0:
        hop = max(1, nperseg // 4)

    # Total frames if we did *all* hops
    n_frames_full = 1 + max(0, (N - nperseg) // hop)
    # Skip factor to cap time columns
    skip = max(1, int(math.ceil(n_frames_full / float(max_cols))))

    # Conservative estimate for final size
    T_est = int(math.ceil(n_frames_full / skip))
    F = nperseg

    # Memory safety check for waterfall size
    estimated_mb = (F * T_est * 4) / (1024 * 1024)  # float32 = 4 bytes
    if estimated_mb > 1024:  # 1GB limit for waterfall
        # Reduce max_cols to stay within memory limits
        max_cols = max(500, int(max_cols * 1024 / estimated_mb))
        skip = max(1, int(math.ceil(n_frames_full / float(max_cols))))
        T_est = int(math.ceil(n_frames_full / skip))
        print_progress(f"Reducing waterfall columns to {T_est} to prevent memory crash")

    # Pre-allocate output array instead of growing list
    S_db = np.zeros((F, T_est), dtype=np.float32)
    t_cols = np.zeros(T_est, dtype=np.float32)

    # Iterate frames with hop*skip
    idx = 0
    frame_i = 0
    col_count = 0

    while idx + nperseg <= N and col_count < T_est:
        if (frame_i % skip) == 0:
            fr = x[idx:idx + nperseg]  # complex64
            frw = (fr * win).astype(np.complex64, copy=False)

            # FFT as complex64; pocketfft preserves precision of input
            W = np.fft.fft(frw, n=nperseg)
            # magnitude in dB (float32)
            mag = np.abs(W).astype(np.float32, copy=False)
            # shift frequency to [-fs/2, fs/2)
            mag = np.fft.fftshift(mag)
            # dB scale
            col_db = 20.0 * np.log10(np.maximum(mag, 1e-12)).astype(np.float32, copy=False)

            # Store directly in pre-allocated array
            S_db[:, col_count] = col_db
            t_cols[col_count] = idx / float(fs)
            col_count += 1

        frame_i += 1
        idx += hop

    if col_count == 0:
        # Edge case: too short
        f = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / fs)).astype(np.float32)
        return f, np.array([], dtype=np.float32), np.zeros((nperseg, 0), dtype=np.float32)

    # Trim to actual size used
    S_db = S_db[:, :col_count]
    t_vis = t_cols[:col_count]

    # Frequency axis (shifted)
    f = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / fs)).astype(np.float32)

    return f, t_vis, S_db


def noise_floor_db(psd_db):
    # robust floor: median over frequency
    return float(np.median(psd_db))

def find_carriers(f, psd_db, floor_db, min_prom_db=8.0, min_bw_bins=3):
    peaks, props = sig.find_peaks(psd_db, prominence=min_prom_db)
    carriers = []
    if len(f) < 2:
        return carriers
    df = float(f[1] - f[0])
    for p in peaks:
        pk_db = float(psd_db[p])
        thr = pk_db - 6.0  # -6 dB width
        l = p
        while l > 0 and psd_db[l] > thr: l -= 1
        r = p
        while r < len(psd_db)-1 and psd_db[r] > thr: r += 1
        if (r - l) < min_bw_bins: 
            continue
        carriers.append({
            "f_center_hz": float(f[p]),
            "p_db": pk_db,
            "bw_hz_est": float((r - l) * df),
            "snr_db_est": float(pk_db - floor_db),
        })
    return carriers

def cfar_mask_chunked(S_db, guard=1, train=6, k=3.0, max_chunk_size=1000):
    """
    Optimized 2D CFAR with chunked processing for large waterfalls.
    S_db: [F, T] dB
    """
    F, T = S_db.shape

    # For small waterfalls, use original algorithm
    if F * T < 1e6:
        return cfar_mask_original(S_db, guard, train, k)

    print_progress(f"Computing CFAR mask ({F}x{T}) with chunked processing...")
    start_time = time.time()

    # Chunked time-axis CFAR
    mask_t = np.zeros_like(S_db, dtype=bool)
    t_chunks = max(1, T // max_chunk_size)
    t_chunk_size = T // t_chunks

    for chunk_i in range(t_chunks):
        t_start = chunk_i * t_chunk_size
        t_end = min(T, (chunk_i + 1) * t_chunk_size)

        # Extend chunk boundaries for context
        t_start_ext = max(0, t_start - train - guard)
        t_end_ext = min(T, t_end + train + guard)

        chunk_data = S_db[:, t_start_ext:t_end_ext]
        chunk_mask = np.zeros((F, t_end - t_start), dtype=bool)

        # Process time-CFAR for this chunk
        for t_rel in range(t_end - t_start):
            t_abs = t_start + t_rel
            t_chunk_rel = t_abs - t_start_ext

            t0 = max(0, t_chunk_rel - train - guard)
            t1 = max(0, t_chunk_rel - guard)
            t2 = min(chunk_data.shape[1], t_chunk_rel + guard + 1)
            t3 = min(chunk_data.shape[1], t_chunk_rel + guard + 1 + train)

            if t1 - t0 > 0 and t3 - t2 > 0:
                ref = np.concatenate([chunk_data[:, t0:t1], chunk_data[:, t2:t3]], axis=1)
                if ref.size > 0:
                    mu = np.median(ref, axis=1, keepdims=True)
                    mad = np.median(np.abs(ref - mu), axis=1, keepdims=True) + 1e-12
                    thr = mu + k * 1.4826 * mad
                    chunk_mask[:, t_rel] = (chunk_data[:, t_chunk_rel:t_chunk_rel+1] > thr).flatten()

        mask_t[:, t_start:t_end] = chunk_mask

        if chunk_i % 10 == 0:  # Progress every 10 chunks
            elapsed = time.time() - start_time
            progress = (chunk_i + 1) / t_chunks
            print_progress(f"Time-CFAR: {progress*100:.0f}% ({chunk_i+1}/{t_chunks})", elapsed)

    # Chunked frequency-axis CFAR
    mask_f = np.zeros_like(S_db, dtype=bool)
    f_chunks = max(1, F // max_chunk_size)
    f_chunk_size = F // f_chunks

    for chunk_i in range(f_chunks):
        f_start = chunk_i * f_chunk_size
        f_end = min(F, (chunk_i + 1) * f_chunk_size)

        # Extend chunk boundaries for context
        f_start_ext = max(0, f_start - train - guard)
        f_end_ext = min(F, f_end + train + guard)

        chunk_data = S_db[f_start_ext:f_end_ext, :]
        chunk_mask = np.zeros((f_end - f_start, T), dtype=bool)

        # Process freq-CFAR for this chunk
        for f_rel in range(f_end - f_start):
            f_abs = f_start + f_rel
            f_chunk_rel = f_abs - f_start_ext

            f0 = max(0, f_chunk_rel - train - guard)
            f1 = max(0, f_chunk_rel - guard)
            f2 = min(chunk_data.shape[0], f_chunk_rel + guard + 1)
            f3 = min(chunk_data.shape[0], f_chunk_rel + guard + 1 + train)

            if f1 - f0 > 0 and f3 - f2 > 0:
                ref = np.concatenate([chunk_data[f0:f1, :], chunk_data[f2:f3, :]], axis=0)
                if ref.size > 0:
                    mu = np.median(ref, axis=0, keepdims=True)
                    mad = np.median(np.abs(ref - mu), axis=0, keepdims=True) + 1e-12
                    thr = mu + k * 1.4826 * mad
                    chunk_mask[f_rel, :] = (chunk_data[f_chunk_rel:f_chunk_rel+1, :] > thr).flatten()

        mask_f[f_start:f_end, :] = chunk_mask

        if chunk_i % 10 == 0:  # Progress every 10 chunks
            elapsed = time.time() - start_time
            progress = (chunk_i + 1) / f_chunks
            print_progress(f"Freq-CFAR: {progress*100:.0f}% ({chunk_i+1}/{f_chunks})", elapsed)

    elapsed = time.time() - start_time
    print_progress(f"CFAR mask completed", elapsed)
    return mask_t & mask_f

def cfar_mask_original(S_db, guard=1, train=6, k=3.0):
    """
    Original 2D CFAR implementation for small datasets.
    S_db: [F, T] dB
    """
    F, T = S_db.shape
    mask_t = np.zeros_like(S_db, dtype=bool)
    for t in range(T):
        t0 = max(0, t - train - guard); t1 = max(0, t - guard)
        t2 = min(T, t + guard + 1);     t3 = min(T, t + guard + 1 + train)
        ref = np.concatenate([S_db[:, t0:t1], S_db[:, t2:t3]], axis=1)
        if ref.size == 0: continue
        mu = np.median(ref, axis=1, keepdims=True)
        mad = np.median(np.abs(ref - mu), axis=1, keepdims=True) + 1e-12
        thr = mu + k * 1.4826 * mad
        mask_t[:, t] = (S_db[:, t:t+1] > thr).flatten()
    mask_f = np.zeros_like(S_db, dtype=bool)
    for f_i in range(F):
        f0 = max(0, f_i - train - guard); f1 = max(0, f_i - guard)
        f2 = min(F, f_i + guard + 1);     f3 = min(F, f_i + guard + 1 + train)
        ref = np.concatenate([S_db[f0:f1, :], S_db[f2:f3, :]], axis=0)
        if ref.size == 0: continue
        mu = np.median(ref, axis=0, keepdims=True)
        mad = np.median(np.abs(ref - mu), axis=0, keepdims=True) + 1e-12
        thr = mu + k * 1.4826 * mad
        mask_f[f_i, :] = (S_db[f_i:f_i+1, :] > thr).flatten()
    return mask_t & mask_f

# Alias for backward compatibility
cfar_mask = cfar_mask_chunked

def save_png(path, fig):
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="Path to I/Q .wav (stereo)")
    ap.add_argument("--out", default="out_report", help="Output folder")
    ap.add_argument("--fs_hint", type=float, default=None, help="Override WAV header Fs (Hz)")
    ap.add_argument("--conv", default="I+Q", choices=["I+Q","I-Q","Q+I"],
                    help="I/Q convention if Step-1 told you to swap/flip")
    ap.add_argument("--nperseg", type=int, default=4096, help="FFT window length")
    ap.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction [0..0.9]")
    ap.add_argument("--prom_db", type=float, default=8.0, help="Min prominence (dB) for PSD peaks")
    ap.add_argument("--cfar_k", type=float, default=3.0, help="CFAR k (higher = stricter)")
    ap.add_argument("--cfar_guard", type=int, default=1, help="Guard cells per axis")
    ap.add_argument("--cfar_train", type=int, default=6, help="Training cells per axis")
    # New optimization parameters
    ap.add_argument("--max_duration", type=float, default=60.0, help="Max processing duration (s) for large files")
    ap.add_argument("--force_full", action="store_true", help="Force full processing (disable sampling)")
    ap.add_argument("--adaptive_sampling", action="store_true", help="Use adaptive sampling strategy to optimize IRR")
    ap.add_argument("--step1_dir", default="out_report", help="Step 1 output directory for IRR comparison")
    args = ap.parse_args()

    overall_start = time.time()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Load WAV metadata first
    info = sf.info(args.wav)
    fs_hdr = info.samplerate
    fs = float(args.fs_hint) if args.fs_hint else float(fs_hdr)
    N_total = info.frames
    C = info.channels

    if C < 2:
        raise SystemExit(f"Expected stereo I/Q. Got {C} channel(s).")

    total_duration = N_total / fs
    print_progress(f"File: {args.wav}")
    print_progress(f"Total: {N_total/1e6:.1f}M samples, {total_duration:.1f}s @ {fs/1e6:.1f} MHz")

    # Determine if we need sampling
    duration = N_total / fs
    need_sampling = duration > args.max_duration and not args.force_full and N_total > 10e6

    # For adaptive sampling, we need a small data sample to determine the best I/Q convention first
    use_sampling = False
    sample_indices = None
    effective_fs = fs
    reduction_factor = 1.0
    chosen_sampling_factor = 1

    if need_sampling:
        if args.adaptive_sampling:
            # Step 1: Load a small sample to determine optimal I/Q convention
            print_progress(f"Large file detected: {duration:.1f}s ({N_total/1e6:.1f}M samples)")
            print_progress("Loading sample for I/Q convention analysis...")

            # Load first 5M samples for convention analysis
            sample_size = min(5_000_000, N_total)
            x_sample, _ = sf.read(args.wav, frames=sample_size, always_2d=True)
            x_sample = x_sample.astype(np.float32, copy=False)

            if x_sample.shape[1] < 2:
                raise SystemExit(f"Expected stereo I/Q. Got {x_sample.shape[1]} channel(s).")

            I_sample = x_sample[:, 0]
            Q_sample = x_sample[:, 1]

            # Find best convention quickly
            conventions = ["I+jQ", "I-jQ", "Q+jI", "Q-jI"]
            best_conv = "I+jQ"  # default
            best_irr = float('-inf')

            for conv in conventions:
                try:
                    if conv == "I+jQ":
                        xc_test = (I_sample - np.mean(I_sample)) + 1j * (Q_sample - np.mean(Q_sample))
                    elif conv == "I-jQ":
                        xc_test = (I_sample - np.mean(I_sample)) - 1j * (Q_sample - np.mean(Q_sample))
                    elif conv == "Q+jI":
                        xc_test = (Q_sample - np.mean(Q_sample)) + 1j * (I_sample - np.mean(I_sample))
                    else:  # Q-jI
                        xc_test = (Q_sample - np.mean(Q_sample)) - 1j * (I_sample - np.mean(I_sample))

                    irr_test = calculate_irr_db(xc_test)
                    if irr_test > best_irr:
                        best_irr = irr_test
                        best_conv = conv
                except:
                    continue

            print_progress(f"Sample analysis: Best convention {best_conv} (IRR: {best_irr:+.1f} dB)")

            # Step 2: Use adaptive sampling with the best convention
            use_sampling, sample_indices, effective_fs, reduction_factor, chosen_sampling_factor = adaptive_sampling_strategy(
                N_total, fs, I_sample, Q_sample, best_conv, target_duration=args.max_duration
            )
        else:
            # Use legacy smart sampling strategy with IRR-friendly improvements
            use_sampling, sample_indices, effective_fs, reduction_factor = smart_sampling_strategy(
                N_total, fs, target_duration=args.max_duration
            )
            chosen_sampling_factor = int(reduction_factor)

    if args.force_full:
        use_sampling = False
        effective_fs = fs
        reduction_factor = 1.0
        chosen_sampling_factor = 1
        print_progress("--force_full: Processing entire file")

    # Estimate memory usage and check safety
    nperseg = int(args.nperseg)
    noverlap = int(nperseg * args.overlap)
    N_eff = len(sample_indices) if use_sampling else N_total
    mem_est_mb, n_windows = estimate_memory_usage(N_eff, nperseg, args.overlap)
    print_progress(f"Estimated memory: {mem_est_mb:.1f} MB for {n_windows} time windows")

    # Check memory safety before proceeding
    try:
        check_memory_safety(mem_est_mb)
    except RuntimeError as e:
        print_progress(f"MEMORY SAFETY ERROR: {e}")
        if not args.force_full:
            raise SystemExit("Aborting to prevent system crash. Use --max_duration for smaller processing window.")

    # Load data with optimized sampling if needed
    load_start = time.time()
    if use_sampling:
        # Use chunked reading for memory efficiency with large files
        x = read_with_chunked_sampling(args.wav, chosen_sampling_factor)
    else:
        x, _ = sf.read(args.wav, always_2d=True)
        x = x.astype(np.float32, copy=False)

    N, C = x.shape
    print_progress(f"Loaded {N/1e6:.1f}M samples", time.time() - load_start)

    # Complex IQ conversion
    iq_start = time.time()
    xc = to_complex_iq(x, conv=args.conv)
    xc = dc_remove(xc)
    print_progress(f"I/Q conversion completed", time.time() - iq_start)

    # IRR Analysis and Comparison with Step 1
    irr_start = time.time()

    # Read Step 1 IRR data if available
    step1_irr = read_step1_irr(args.step1_dir)

    # Calculate corrected IRR after applying --conv
    corrected_irr_db = calculate_irr_db(xc)
    print_progress(f"IRR calculation completed", time.time() - irr_start)

    # IRR comparison and reporting
    if step1_irr:
        step1_irr_db = step1_irr.get("step1_best_irr_db", 0.0)
        step1_convention = step1_irr.get("step1_best_convention", "Unknown")
        irr_improvement_db = corrected_irr_db - step1_irr_db

        print_progress(f"IRR Comparison:")
        print_progress(f"  Step 1 (best): {step1_irr_db:.2f} dB ({step1_convention})")
        print_progress(f"  Step 2 (corrected): {corrected_irr_db:.2f} dB (--conv {args.conv})")
        print_progress(f"  Improvement: {irr_improvement_db:+.2f} dB")

        if irr_improvement_db > 5.0:
            print_progress(f"[OK] Significant IRR improvement - correction is effective!")
        elif irr_improvement_db > 1.0:
            print_progress(f"[OK] Moderate IRR improvement")
        elif irr_improvement_db > -1.0:
            print_progress(f"[WARN] Minimal change in IRR")
        else:
            print_progress(f"[ERROR] IRR degraded - check --conv parameter")
    else:
        step1_irr_db = None
        step1_convention = "Unknown"
        irr_improvement_db = None
        print_progress(f"Corrected IRR: {corrected_irr_db:.2f} dB (--conv {args.conv})")
        print_progress(f"[WARN] No Step 1 data found for comparison")

    # PSD (Welch) - always compute on effective data
    psd_start = time.time()
    f_psd, Pxx = welch_psd(xc, effective_fs, nperseg, noverlap)
    Pxx_db = 10*np.log10(np.maximum(Pxx, 1e-20))
    floor_db = noise_floor_db(Pxx_db)
    carriers = find_carriers(f_psd, Pxx_db, floor_db, min_prom_db=args.prom_db)
    print_progress(f"PSD analysis: found {len(carriers)} carriers", time.time() - psd_start)

    # Waterfall (STFT)
    stft_start = time.time()
    f, t, S_db = stft_waterfall(xc, effective_fs, nperseg, noverlap)
    print_progress(f"STFT waterfall ({S_db.shape[0]}x{S_db.shape[1]})", time.time() - stft_start)
    # Build a visualization thumbnail to avoid huge RGBA allocations in Matplotlib
    # Keep frequency full, cap time to ~4000 columns. Use max to preserve bursts.
    S_db_vis, f_vis, t_vis = make_waterfall_thumbnail(S_db, f, t, max_cols=4000, max_rows=None, op="max")
    print_progress(f"Waterfall thumbnail for plotting: {S_db_vis.shape[0]}x{S_db_vis.shape[1]}")

    # CFAR activity mask + occupancy
    cfar_start = time.time()
    mask = cfar_mask(S_db, guard=args.cfar_guard, train=args.cfar_train, k=args.cfar_k)
    occ = mask.mean(axis=1)  # per-frequency occupancy [0..1]
    time_burstiness = mask.mean(axis=0)
    print_progress(f"CFAR detection completed", time.time() - cfar_start)

    # -------- Save figures --------
    plot_start = time.time()

    # PSD
    fig = plt.figure(figsize=(10,4))
    plt.plot(f_psd/1e6, Pxx_db, lw=1)
    plt.axhline(floor_db, ls="--", alpha=0.5, label="noise floor")
    for c in carriers:
        plt.axvline(c["f_center_hz"]/1e6, color="r", alpha=0.3)
    title = f"Welch PSD"
    if use_sampling:
        title += f" (1:{int(reduction_factor)} sampled)"
    plt.title(title); plt.xlabel("Frequency (MHz)"); plt.ylabel("dB/Hz")
    plt.legend(loc="lower left")
    save_png(Path(args.out, "psd.png"), fig)

    # Waterfall (thumbnail)
    fig = plt.figure(figsize=(10,5))
    extent = [f_vis[0]/1e6, f_vis[-1]/1e6, t_vis[-1], t_vis[0]]
    plt.imshow(S_db_vis, aspect="auto", extent=extent, cmap="viridis", origin="upper")
    plt.colorbar(label="dB")
    title = "Waterfall (STFT magnitude, dB)"
    if 'use_sampling' in locals() and use_sampling:
        title += f" - 1:{int(reduction_factor)} sampled"
    if S_db_vis.shape[1] < (t.size if isinstance(t, np.ndarray) else S_db.shape[1]):
        title += f" - downsampled to {S_db_vis.shape[1]} cols"
    plt.title(title)
    plt.xlabel("Frequency (MHz)"); plt.ylabel("Time (s)")
    save_png(Path(args.out, "waterfall.png"), fig)

    # Occupancy
    fig = plt.figure(figsize=(10,3))
    plt.plot(f/1e6, occ, lw=1)
    plt.ylim(0, 1.02)
    title = f"Occupancy (CFAR activity fraction)"
    if use_sampling:
        title += f" - 1:{int(reduction_factor)} sampled"
    plt.title(title); plt.xlabel("Frequency (MHz)"); plt.ylabel("Duty cycle")
    save_png(Path(args.out, "occupancy.png"), fig)

    print_progress(f"Generated plots", time.time() - plot_start)

    # -------- Save CSVs / JSON --------
    io_start = time.time()

    with open(Path(args.out,"carriers.csv"), "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=["f_center_hz","p_db","bw_hz_est","snr_db_est"])
        w.writeheader()
        for c in carriers:
            w.writerow(c)

    np.savetxt(Path(args.out,"time_activity.csv"),
               np.c_[t, time_burstiness], delimiter=",",
               header="time_s,activity", comments="")

    # Enhanced summary with optimization info and IRR comparison
    effective_duration = N / effective_fs
    summary = {
        "wav": str(Path(args.wav).resolve()),
        "fs_header_hz": float(fs_hdr),
        "fs_used_hz": float(effective_fs),
        "original_fs_hz": float(fs),
        "duration_s": float(effective_duration),
        "original_duration_s": float(total_duration),
        "samples_processed": int(N),
        "original_samples": int(N_total),
        "nperseg": nperseg,
        "overlap": float(args.overlap),
        "noise_floor_db": float(floor_db),
        "num_carriers": len(carriers),
        "cfar": {"k": float(args.cfar_k), "guard": int(args.cfar_guard), "train": int(args.cfar_train)},
        "prom_db": float(args.prom_db),
        "optimization": {
            "used_sampling": use_sampling,
            "reduction_factor": float(reduction_factor),
            "target_duration_s": float(args.max_duration),
            "memory_estimate_mb": float(mem_est_mb),
            "waterfall_shape": [int(S_db.shape[0]), int(S_db.shape[1])]
        },
        "irr_analysis": {
            "conv_applied": args.conv,
            "corrected_irr_db": float(corrected_irr_db),
            "step1_irr_db": float(step1_irr_db) if step1_irr_db is not None else None,
            "step1_convention": step1_convention,
            "irr_improvement_db": float(irr_improvement_db) if irr_improvement_db is not None else None,
            "step1_file": step1_irr.get("step1_file", None) if step1_irr else None
        }
    }
    Path(args.out, "summary.json").write_text(json.dumps(summary, indent=2))

    print_progress(f"Saved outputs", time.time() - io_start)

    total_elapsed = time.time() - overall_start
    print_progress(f"Analysis completed in {total_elapsed:.1f}s")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

#python analyze_iq_unknown.py your_20MHz.wav --fs_hint 20000000 --out out_report --nperseg 4096 --overlap 0.5 --prom_db 8 --cfar_k 3.0

#If the PSD marks too many tiny peaks, raise --prom_db to 10–12.
#If the waterfall mask is too “speckly”, raise --cfar_k to 3.5–4.0 (stricter).
#If Step-1 told you to swap/flip I/Q, add --conv I-Q or --conv Q+I.

# analyze_iq_unknown.py — Step 2: PSD, waterfall, occupancy, carrier table
import argparse, json, csv
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal as sig
import matplotlib.pyplot as plt
import time
import sys

# ---------- Utils ----------
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

    total_mb = (stft_memory + waterfall_memory) / (1024 * 1024)
    return total_mb, n_windows

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

def stft_waterfall(xc, fs, nperseg, noverlap):
    f, t, Z = sig.stft(
        xc, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
        return_onesided=False, boundary=None, detrend=False
    )
    f = np.fft.fftshift(f)
    Z = np.fft.fftshift(Z, axes=0)
    S_db = 20*np.log10(np.maximum(np.abs(Z), 1e-12))
    return f, t, S_db

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

    print_progress(f"Computing CFAR mask ({F}×{T}) with chunked processing...")
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

    # Determine sampling strategy
    use_sampling, sample_indices, effective_fs, reduction_factor = smart_sampling_strategy(
        N_total, fs, target_duration=args.max_duration
    )

    if args.force_full:
        use_sampling = False
        effective_fs = fs
        reduction_factor = 1.0
        print_progress("--force_full: Processing entire file")

    # Estimate memory usage
    nperseg = int(args.nperseg)
    noverlap = int(nperseg * args.overlap)
    N_eff = len(sample_indices) if use_sampling else N_total
    mem_est_mb, n_windows = estimate_memory_usage(N_eff, nperseg, args.overlap)
    print_progress(f"Estimated memory: {mem_est_mb:.1f} MB for {n_windows} time windows")

    # Load data with sampling if needed
    load_start = time.time()
    if use_sampling:
        # Use chunked reading for memory efficiency with large files
        x = read_with_chunked_sampling(args.wav, reduction_factor)
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
    print_progress(f"STFT waterfall ({S_db.shape[0]}×{S_db.shape[1]})", time.time() - stft_start)

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

    # Waterfall
    fig = plt.figure(figsize=(10,5))
    extent=[f[0]/1e6, f[-1]/1e6, t[-1], t[0]]
    plt.imshow(S_db, aspect="auto", extent=extent, cmap="viridis")
    plt.colorbar(label="dB")
    title = f"Waterfall (STFT magnitude, dB)"
    if use_sampling:
        title += f" - 1:{int(reduction_factor)} sampled"
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

    # Enhanced summary with optimization info
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
        }
    }
    Path(args.out, "summary.json").write_text(json.dumps(summary, indent=2))

    print_progress(f"Saved outputs", time.time() - io_start)

    total_elapsed = time.time() - overall_start
    print_progress(f"Analysis completed in {total_elapsed:.1f}s")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

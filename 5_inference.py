#!/usr/bin/env python3
"""
Wideband (.r64 / raw-float32 .wav via --force_raw) segmentation + classification at 20 MHz
- Splits gigantic I/Q files into manageable chunk files on disk, then processes chunks.
- VAD -> frame-level inference -> per-burst grouping across chunks.
- For each burst (a contiguous transmission), estimate:
    • start/end time (s) and absolute timestamps (if --recording_start provided)
    • RF center (offset Hz and absolute Hz if --rf_center_hz given)
    • occupied bandwidth (Hz) via -20 dB relative threshold
    • SNR (dB) via robust PSD floor vs in-band power
- Saves one master CSV: predictions.csv (one row per BURST)
- Optionally saves frames.csv for per-frame breakdown and spectrogram PNG per burst

Notes
-----
• If your source is a raw float32 .wav (IEEE float), Python's built-in `wave` module
  does not expose the IEEE-float flag. For such files, use `--force_raw --sr 20000000`
  so the reader treats it as raw interleaved float32 I/Q (.r64-style) for correctness.
• If you DO have a standard PCM WAV header (RIFF/PCM), you can omit --force_raw.
"""

import argparse, os, json, math, wave, csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from contextlib import nullcontext
import itertools
from types import SimpleNamespace

# -------------- IO helpers (raw & wav) --------------
BYTES_PER_COMPLEX = 8  # 2 * float32

def load_r64(path, channel_order="IQ", start=0, end=None):
    size = os.path.getsize(path)
    total_samples = size // BYTES_PER_COMPLEX
    if end is None or end > total_samples:
        end = total_samples
    start = max(0, start)
    if start >= end:
        return np.zeros((2,0), dtype=np.float32)
    mm = np.memmap(path, dtype=np.float32, mode='r', shape=(2*total_samples,))
    raw = np.array(mm[start*2:end*2], copy=False)
    del mm
    I = raw[0::2]; Q = raw[1::2]
    if channel_order.upper() == "QI":
        I, Q = Q, I
    return np.stack([I, Q], axis=0)

def _pcm24_to_float32(buf_bytes):
    a = np.frombuffer(buf_bytes, dtype=np.uint8).reshape(-1, 3)
    b = (a[:,0].astype(np.int32) | (a[:,1].astype(np.int32) << 8) | (a[:,2].astype(np.int32) << 16))
    neg = b & 0x800000
    b = b - (neg << 1)
    return (b.astype(np.float32) / 8388608.0)

def load_wav_iq(path, channel_order="IQ"):
    try:
        # First try soundfile for RF64/RIFF support
        import soundfile as sf
        data, sr = sf.read(path, always_2d=True, dtype='float32')
        data = data.T  # soundfile returns [time, channels], we want [channels, time]

        if data.shape[0] == 1:
            # Mono file - assume interleaved I/Q
            mono = data[0]
            I = mono[0::2]
            Q = mono[1::2]
        elif data.shape[0] >= 2:
            # Stereo+ file - use first two channels as I/Q
            I = data[0]
            Q = data[1]
        else:
            raise ValueError("No audio channels found")

        if channel_order.upper() == "QI":
            I, Q = Q, I

        return np.stack([I, Q], axis=0), sr

    except ImportError:
        # Fallback to wave module if soundfile not available
        with wave.open(path, "rb") as wf:
            nchan = wf.getnchannels(); sampwidth = wf.getsampwidth(); sr = wf.getframerate(); nframes = wf.getnframes()
            raw = wf.readframes(nframes)
        if sampwidth == 1:
            x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32); x = (x - 128.0) / 127.5
        elif sampwidth == 2:
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 3:
            x = _pcm24_to_float32(raw)
        elif sampwidth == 4:
            # Many SDR WAVs are IEEE float32; the wave module doesn't expose the format code.
            # Treat as raw float32 samples in range [-1,1].
            x = np.frombuffer(raw, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth*8} bits")
        if nchan == 1:
            I = x[0::2]; Q = x[1::2]
            if channel_order.upper() == "QI": I, Q = Q, I
            return np.stack([I, Q], axis=0), sr
        x = x.reshape(-1, nchan).T
        I = x[0]; Q = x[1] if nchan >= 2 else np.zeros_like(I)
        if channel_order.upper() == "QI": I, Q = Q, I
        return np.stack([I, Q], axis=0), sr

# -------------- Framing / basic math --------------

def frame_gen(x, T, stride):
    tlen = x.shape[1]
    if tlen <= 0: return
    last = max(0, tlen - T)
    for s in range(0, last + 1, stride):
        yield s, x[:, s:s+T]

def energy_db(frame):
    f = np.asarray(frame, dtype=np.float64)
    e = np.mean(f*f)
    if not np.isfinite(e): e = 0.0
    return 10*np.log10(max(e, 1e-12))

def softmax_np(logits):
    m = np.max(logits); ex = np.exp(logits - m)
    return ex / np.sum(ex)

# -------------- Robust normalization & enriched channels --------------

def safe_normalize_iq(I, Q):
    """Robust IQ normalization that avoids overflow warnings.
    - Works in float64 for the divide, then clips and casts back to float32.
    - Floors the scale to a small epsilon to avoid huge quotients when signal is near-zero.
    """
    I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    abs_vals = np.maximum(np.abs(I), np.abs(Q))
    if not np.any(abs_vals):
        return I, Q

    # Robust per-chunk scale from 99.9th percentile
    scale = float(np.percentile(abs_vals, 99.9))
    # Guard against tiny scales (near-silence / denormals)
    SCALE_FLOOR = 1e-6
    if not np.isfinite(scale) or scale < SCALE_FLOOR:
        m = float(np.max(abs_vals))
        scale = m if (np.isfinite(m) and m >= SCALE_FLOOR) else 1.0

    # Do the division in float64 to prevent float32 overflow, then clip & cast back
    I64 = I.astype(np.float64, copy=False)
    Q64 = Q.astype(np.float64, copy=False)
    I = np.clip(I64/scale, -4.0, 4.0).astype(np.float32, copy=False)
    Q = np.clip(Q64/scale, -4.0, 4.0).astype(np.float32, copy=False)
    return I, Q


def enrich_channels(x2, in_ch):
    I, Q = x2.astype(np.float32, copy=False)
    I, Q = safe_normalize_iq(I, Q)
    amp = np.hypot(I.astype(np.float64), Q.astype(np.float64)).astype(np.float32)
    phase = np.arctan2(Q.astype(np.float64), I.astype(np.float64)).astype(np.float32)
    dI = np.diff(I, prepend=I[:1]).astype(np.float32); dQ = np.diff(Q, prepend=Q[:1]).astype(np.float32)
    damp = np.diff(amp, prepend=amp[:1]).astype(np.float32)
    ph_unw = np.unwrap(phase.astype(np.float64)).astype(np.float32)
    dphase = np.diff(ph_unw, prepend=ph_unw[:1]).astype(np.float32)
    def movavg(x, k=5):
        x = x.astype(np.float64, copy=False)
        if x.size < k: return x.astype(np.float32, copy=False)
        c = np.cumsum(np.pad(x, (1,0), mode="edge"))
        y = (c[k:] - c[:-k]) / float(k)
        head = np.full(k//2, y[0], dtype=np.float64); tail = np.full(k - k//2 - 1, y[-1], dtype=np.float64)
        return np.concatenate([head, y, tail]).astype(np.float32, copy=False)
    I_ma = movavg(I); Q_ma = movavg(Q); amp_ma = movavg(amp)
    # Create additional derived features to get proper 14 channels
    d2I = np.diff(dI, prepend=dI[:1]).astype(np.float32)  # second derivative of I
    d2Q = np.diff(dQ, prepend=dQ[:1]).astype(np.float32)  # second derivative of Q
    base14 = [I, Q, amp, phase, dI, dQ, damp, dphase, I_ma, Q_ma, amp_ma, ph_unw, d2I, d2Q]
    base14 = [np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False) for ch in base14]
    X = np.stack(base14, axis=0)
    if in_ch <= X.shape[0]:
        return X[:in_ch]
    reps = int(np.ceil(in_ch / X.shape[0])); return np.tile(X, (reps,1))[:in_ch]

# -------------- VAD --------------

def vad_detect(x, sr, T, stride, thr_db, hang_db=3.0, min_dur_ms=30, merge_gap_ms=30):
    start_thr = thr_db; stop_thr = thr_db - abs(hang_db)
    min_dur = int((min_dur_ms/1000.0)*sr); merge_gap = int((merge_gap_ms/1000.0)*sr)
    starts, ends, on, open_start = [], [], False, None
    for s, fr in frame_gen(x, T, stride):
        e = energy_db(fr)
        if not on and e >= start_thr:
            on = True; open_start = s
        elif on and e <= stop_thr:
            on = False; seg_start = open_start; seg_end = s + T
            if seg_end - seg_start >= min_dur:
                starts.append(seg_start); ends.append(seg_end)
            open_start = None
    if on and open_start is not None:
        seg_start = open_start; seg_end = x.shape[1]
        if seg_end - seg_start >= min_dur:
            starts.append(seg_start); ends.append(seg_end)
    if not starts: return []
    merged = []; cs, ce = starts[0], ends[0]
    for s, e in zip(starts[1:], ends[1:]):
        if s - ce <= merge_gap: ce = e
        else: merged.append((cs, ce)); cs, ce = s, e
    merged.append((cs, ce))
    return merged

# -------------- Spectrograms --------------

def make_spectrogram_png(sig_iq, sr, out_path, nfft=2048, hop=256):
    I, _ = sig_iq; n = len(I)
    if n <= 0:
        plt.figure(figsize=(6,3)); plt.title("Empty segment"); plt.savefig(out_path); plt.close(); return
    win = np.hanning(nfft)
    pad = (math.ceil(max(0, n - nfft)/hop) * hop + nfft) - n if n >= nfft else nfft - n
    Ipad = np.pad(I, (0, pad))
    frames = np.lib.stride_tricks.sliding_window_view(Ipad, nfft)[::hop]
    if len(frames) == 0: S = np.zeros((1, nfft//2+1), dtype=np.float32)
    else:
        S = np.abs(np.fft.rfft(frames*win, axis=-1))
        S = 20*np.log10(np.maximum(S, 1e-12))
    plt.figure(figsize=(6,3))
    plt.imshow(S.T, origin="lower", aspect="auto")
    plt.title("Spectrogram (I mag)"); plt.xlabel("frame"); plt.ylabel("freq bin")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

# -------------- RF metrics (center / BW / SNR) --------------

def welch_psd_streaming(x,
                        sr,
                        nfft=4096,
                        hop=1024,
                        max_frames=8192,
                        preview_sr=None,
                        window=np.hanning):
    """
    Memory-safe Welch-like PSD:
    - Optionally downsample x for PSD preview only (preview_sr)
    - Iterate frames in a loop and accumulate power; no giant 2-D array
    - Limit frames processed to max_frames by skipping strides
    Returns (freqs_normalized, P_db)
    """
    # Optional preview downsample for PSD only (keep center/bw estimates stable)
    if preview_sr is not None and preview_sr > 0 and sr > preview_sr:
        step = int(round(sr / float(preview_sr)))
        if step > 1:
            x = x[::step]
            sr = sr / step

    N = x.size
    if N <= 0:
        return np.linspace(-0.5, 0.5, nfft, endpoint=False), np.full(nfft, -200.0, dtype=np.float64)

    # Ensure types are light and normalize to prevent overflow
    x = np.asarray(x, dtype=np.complex64, order="C")
    # Normalize I/Q data to prevent FFT overflow
    x_max = np.max(np.abs(x))
    if x_max > 1e6:  # Scale down very large I/Q values
        x = x / x_max * 1e3
    win = np.asarray(window(nfft), dtype=np.float32)
    win_energy = float(np.sum(win.astype(np.float64)**2) + 1e-12)

    # Frame indexing
    if N < nfft:
        pad = nfft - N
        x = np.pad(x, (0, pad))
        N = x.size

    n_frames_total = 1 + (N - nfft) // hop
    # Skip factor to cap frames to max_frames
    skip = max(1, n_frames_total // max_frames) if n_frames_total > max_frames else 1

    P_accum = np.zeros(nfft, dtype=np.float64)
    m = 0
    i = 0
    # Iterate frames with hop*skip
    while i + nfft <= N:
        fr = x[i:i+nfft]
        frw = fr * win
        # Complex64 FFT -> complex64 output in recent NumPy (pocketfft preserves precision)
        W = np.fft.fft(frw, n=nfft)
        # Accumulate power spectrum with overflow protection
        W_abs = np.abs(W).astype(np.float64, copy=False)  # Convert to float64 before squaring
        P_accum += W_abs * W_abs  # Avoid overflow in squaring operation
        m += 1
        i += hop * skip

    if m == 0:
        return np.linspace(-0.5, 0.5, nfft, endpoint=False), np.full(nfft, -200.0, dtype=np.float64)

    # Mean power, shift to [-0.5, 0.5)
    P = P_accum / (m * win_energy)
    P = np.fft.fftshift(P)
    P_db = 10.0 * np.log10(np.maximum(P, 1e-20))
    freqs_n = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0))  # normalized [-0.5,0.5)

    return freqs_n, P_db


def estimate_center_and_bw_and_snr(I, Q, sr,
                                   psd_nfft=4096,
                                   psd_hop=1024,
                                   psd_max_frames=8192,
                                   psd_preview_sr=None):
    """
    Robust center/BW/SNR using streaming PSD to avoid huge allocations.
    """
    x = I.astype(np.float32) + 1j*Q.astype(np.float32)
    freqs_n, P_db = welch_psd_streaming(
        x, sr,
        nfft=psd_nfft,
        hop=psd_hop,
        max_frames=psd_max_frames,
        preview_sr=psd_preview_sr,
        window=np.hanning
    )

    # Peak & -20 dB bandwidth
    k_peak = int(np.argmax(P_db))
    peak_db = float(P_db[k_peak])

    thr = peak_db - 20.0
    above = np.where(P_db >= thr)[0]
    if above.size == 0:
        f0_hz = 0.0
        bw_hz = 0.0
    else:
        f_lo = freqs_n[above[0]]
        f_hi = freqs_n[above[-1]]
        f0_hz = float(((f_lo + f_hi) / 2.0) * sr)
        bw_hz = float((f_hi - f_lo) * sr)

    # Robust floor via trimmed median
    sorted_db = np.sort(P_db)
    trim = int(0.2 * sorted_db.size)
    floor_db = float(np.median(sorted_db[trim:sorted_db.size - trim])) if sorted_db.size > 2 * trim else float(np.median(sorted_db))
    snr_db = float(peak_db - floor_db)
    return f0_hz, bw_hz, snr_db, peak_db, floor_db

# -------------- Retune & Decimate --------------

def _design_lowpass(num_taps, cutoff_norm):
    # cutoff_norm in (0,1) relative to Nyquist
    n = np.arange(num_taps) - (num_taps-1)/2.0
    h = np.sinc(2*cutoff_norm*(n))
    w = np.hamming(num_taps)
    h = h * w
    h = h / np.sum(h)
    return h.astype(np.float64)

def retune_and_decimate(seg_raw, sr, f0_hz, target_sr, taps=127, passfrac=0.40):
    I, Q = seg_raw
    N = I.size
    if N == 0:
        return seg_raw, sr, 1
    # Mix to baseband
    t = np.arange(N, dtype=np.float64) / float(sr)
    osc = np.exp(-1j * 2.0*np.pi * (f0_hz) * t)
    x = (I.astype(np.float64) + 1j*Q.astype(np.float64)) * osc
    # Integer decimation factor close to sr/target
    D = max(1, int(round(sr / float(target_sr))))
    new_sr = sr / D
    # FIR lowpass cutoff relative to old Nyquist
    cutoff_norm_old = (passfrac * (new_sr/2.0)) / (sr/2.0)
    cutoff_norm_old = min(max(cutoff_norm_old, 0.05), 0.49)
    h = _design_lowpass(taps, cutoff_norm_old)
    # Filter I and Q
    xr = np.real(x); xi = np.imag(x)
    yr = np.convolve(xr, h, mode='same')[::D]
    yi = np.convolve(xi, h, mode='same')[::D]
    return np.stack([yr.astype(np.float32), yi.astype(np.float32)], axis=0), new_sr, D

# -------------- Inference core --------------

def classify_segment(x_seg, sr, labels, model, device, args, run_dir, seg_index, global_offset_s=0.0,
                     save_frames=True, save_spectrograms=False, amp_on=False):
    x = np.nan_to_num(x_seg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    I_g, Q_g = safe_normalize_iq(x[0], x[1])
    x_norm_for_vad = np.stack([I_g, Q_g], axis=0)

    # VAD thresholding with automatic large-file detection
    file_size_gb = x.nbytes / (1024**3)
    use_fast_mode = file_size_gb > 1.0  # Auto-enable fast mode for files > 1GB

    if use_fast_mode:
        print(f"[vad-fast] Large file detected ({file_size_gb:.1f}GB), using fast VAD mode")
        # Use fixed threshold for very large files to avoid processing delays
        thr_dbfs = getattr(args, 'vad_abs_dbfs', -30.0) if hasattr(args, 'vad_abs_dbfs') and args.vad_abs_dbfs is not None else -30.0
    elif args.auto_vad or (hasattr(args, 'vad_abs_dbfs') and args.vad_abs_dbfs is not None):
        I, Q = x_norm_for_vad
        # Base series: amplitude envelope or instantaneous power
        series = np.hypot(I.astype(np.float64), Q.astype(np.float64)) if args.envelope_vad else (I.astype(np.float64)**2 + Q.astype(np.float64)**2)

        # Aggressive fast path for large files: always decimate for VAD analysis
        ds = max(1, int(getattr(args, 'vad_down', 200)))  # Default to 200x downsample
        series_work = series[::ds]
        sr_work = sr / ds

        # Cap analyzed points much more aggressively for large files
        max_points = getattr(args, 'auto_vad_max_points', 100_000)  # Reduced from 2M to 100K
        if series_work.size > max_points:
            step2 = int(math.ceil(series_work.size / float(max_points)))
            series_work = series_work[::step2]
            sr_work = sr_work / step2
            print(f"[vad-opt] Using {series_work.size:,} points for VAD analysis (downsampled from {series.size:,})")

        # Smoothing (on the small series)
        win = max(4, int(sr_work * (args.auto_vad_smooth_ms / 1000.0)))
        if series_work.size >= win:
            ker = np.ones(win, dtype=np.float64) / float(win)
            smoothed = np.convolve(series_work, ker, mode="same")
        else:
            smoothed = series_work

        if hasattr(args, 'vad_abs_dbfs') and args.vad_abs_dbfs is not None:
            thr_dbfs = float(args.vad_abs_dbfs)
        else:
            noise_floor = np.quantile(smoothed, args.auto_vad_quantile)
            thr_lin = max(noise_floor, 1e-12) * (10 ** (args.auto_vad_margin_db / 10.0))
            thr_dbfs = 10*np.log10(thr_lin + 1e-12)
    else:
        thr_dbfs = args.vad_thr_db

    segs = vad_detect(x_norm_for_vad, sr, T=args.vad_T, stride=args.vad_stride,
                      thr_db=thr_dbfs, hang_db=args.vad_hang_db,
                      min_dur_ms=args.min_dur_ms, merge_gap_ms=args.merge_gap_ms)
    if not segs and args.fallback_fullfile:
        segs = [(0, x.shape[1])]

    rows = []
    fcsv_path = run_dir / f"frames_seg{seg_index:04d}.csv"

    for i, (s0, s1) in enumerate(segs):
        seg_raw = x[:, s0:s1]
        # RF metrics per VAD segment
        f0_hz, bw_hz, snr_db, peak_db, floor_db = estimate_center_and_bw_and_snr(
        seg_raw[0], seg_raw[1], sr,
        psd_nfft=getattr(args, "psd_nfft", 4096),
        psd_hop=getattr(args, "psd_hop", 1024),
        psd_max_frames=getattr(args, "psd_max_frames", 8192),
        psd_preview_sr=getattr(args, "psd_preview_sr", None)
    )

        # Optional: retune to baseband and decimate before classification
        x_proc = seg_raw
        sr_proc = sr
        if getattr(args, 'retune_decim', False) and np.isfinite(f0_hz):
            try:
                x_proc, sr_proc, _ = retune_and_decimate(seg_raw, sr, f0_hz, args.target_sr, taps=args.lp_taps, passfrac=args.lp_pass)
            except Exception:
                x_proc = seg_raw
                sr_proc = sr

        chan = enrich_channels(x_proc, args.in_ch)
        if chan.shape[1] < args.T_crop:
            pad = args.T_crop - chan.shape[1]
            chan = np.pad(chan, ((0,0),(0,pad)), mode="constant")

        local_logits = []
        frame_rows = []
        buf, buf_starts = [], []

        def flush_buf():
            nonlocal buf, buf_starts, local_logits, frame_rows
            if not buf: return
            cur_bs = max(1, args.batch_size)
            i0 = 0
            while i0 < len(buf):
                i1 = min(len(buf), i0 + cur_bs)
                xb_np = np.stack(buf[i0:i1], axis=0).astype(np.float32, copy=False)
                xb = torch.from_numpy(xb_np).to(device=device, dtype=torch.float32, non_blocking=True)
                mean = xb.mean(dim=(2,), keepdim=True); std = xb.std(dim=(2,), keepdim=True).clamp_min(1e-6)
                xb = (xb - mean) / std
                amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (device.type == "cuda" and amp_on) else nullcontext()
                try:
                    with torch.inference_mode(), amp_ctx:
                        lg = model(xb).detach().cpu().numpy()
                    for k in range(lg.shape[0]):
                        local_logits.append(lg[k])
                        if save_frames:
                            probs = softmax_np(lg[k]); top1 = int(np.argmax(probs))
                            frame_rows.append({
                                "segment_id": seg_index,
                                "frame_start_sample": int(s0 + buf_starts[i0 + k]),
                                "frame_start_time_s": float(global_offset_s + (s0/float(sr)) + (buf_starts[i0 + k]/float(sr_proc))),
                                "top1_idx": top1,
                                "top1_label": labels[top1] if top1 < len(labels) else f"class_{top1}",
                                "top1_prob": float(probs[top1])
                            })
                    i0 = i1
                    if getattr(args, 'cuda_gc', False) and device.type == 'cuda':
                        try: torch.cuda.empty_cache()
                        except Exception: pass
                except RuntimeError as e:
                    msg = str(e).lower()
                    if 'out of memory' in msg and getattr(args, 'auto_batch', False) and cur_bs > max(1, args.min_batch):
                        # halve batch and retry this slice
                        if device.type == 'cuda':
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        cur_bs = max(args.min_batch, cur_bs // 2)
                        continue
                    raise
            buf, buf_starts = [], []


        for ls, fr in frame_gen(chan, args.T_crop, args.stride):
            buf.append(fr.astype(np.float32, copy=False)); buf_starts.append(ls)
            if len(buf) >= max(1, args.batch_size):
                flush_buf()
        flush_buf()
        if not local_logits: continue
        logits_mean = np.mean(np.stack(local_logits, axis=0), axis=0)
        probs_mean = softmax_np(logits_mean); pred_idx = int(np.argmax(probs_mean))
        pred_p = float(probs_mean[pred_idx]); label = labels[pred_idx] if pred_idx < len(labels) else f"class_{pred_idx}"
        # unknown gating
        if args.unknown_by_conf > 0 and pred_p < args.unknown_by_conf: label = "UNKNOWN"
        t0 = global_offset_s + s0 / float(sr); t1 = global_offset_s + s1 / float(sr)
        rows.append({
            "segment_id": seg_index,
            "start_sample": int(s0), "end_sample": int(s1),
            "start_time_s": round(t0, 6), "end_time_s": round(t1, 6),
            "duration_s": round((t1 - t0), 6),
            "pred_idx": pred_idx, "pred_label": label, "prob_top1": round(pred_p, 6),
            "rf_offset_hz": round(f0_hz, 1), "bandwidth_hz": round(bw_hz, 1), "snr_db": round(snr_db, 2)
        })
        if save_frames and frame_rows:
            with open(fcsv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
                if not Path(fcsv_path).exists() or os.stat(fcsv_path).st_size == 0:
                    w.writeheader()
                for fr_row in frame_rows: w.writerow(fr_row)
        if save_spectrograms:
            try: make_spectrogram_png(seg_raw, sr, run_dir / f"seg_{seg_index:04d}_{label}.png")
            except Exception: pass
    return rows, (fcsv_path if Path(fcsv_path).exists() and os.stat(fcsv_path).st_size > 0 else None)

# -------------- Burst grouping across chunks --------------

def group_bursts(segments, merge_gap_s=0.06):
    # segments: list of per-segment rows from classify_segment
    if not segments: return []
    segs = sorted(segments, key=lambda r: r["start_time_s"])  # time order
    bursts = []
    cur = None
    for r in segs:
        if cur is None:
            cur = {**r, "burst_id": 0, "_members": [r], "_prob_sum": r["prob_top1"]}
            continue
        gap = r["start_time_s"] - cur["end_time_s"]
        same_label = (r["pred_label"] == cur["pred_label"]) and (r["pred_label"] != "UNKNOWN")
        if same_label and gap <= merge_gap_s:
            # extend burst
            cur["end_time_s"] = max(cur["end_time_s"], r["end_time_s"])  # coalesce
            cur["duration_s"] = round(cur["end_time_s"] - cur["start_time_s"], 6)
            cur["_members"].append(r); cur["_prob_sum"] += r["prob_top1"]
            # combine RF stats via weighted averages (by segment duration)
            w0 = sum(m["end_time_s"] - m["start_time_s"] for m in cur["_members"]) or 1.0
            cur["rf_offset_hz"] = float(np.average([m["rf_offset_hz"] for m in cur["_members"]], weights=[(m["end_time_s"]-m["start_time_s"]) for m in cur["_members"]]))
            cur["bandwidth_hz"] = float(np.max([m["bandwidth_hz"] for m in cur["_members"]]))
            cur["snr_db"] = float(np.median([m["snr_db"] for m in cur["_members"]]))
        else:
            bursts.append(cur); cur = {**r, "burst_id": cur["burst_id"] + 1, "_members": [r], "_prob_sum": r["prob_top1"]}
    if cur is not None: bursts.append(cur)
    # finalize: drop helper keys and compute avg prob
    for b in bursts:
        b["avg_prob_top1"] = round(b.pop("_prob_sum", 0.0) / max(1, len(b.pop("_members", []))), 6)
    # reindex burst_id sequentially
    for i, b in enumerate(bursts): b["burst_id"] = i
    return bursts

def run_on_file(in_path_str, args, labels, model, device, base_root, stamp):
    """
    Process a single input file and return (bursts, run_dir, frames_csv_paths).
    Each file gets its own subdirectory under base_root.
    """
    in_path = Path(in_path_str)
    run_dir = base_root / in_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # ---- Detect type (WAV vs RAW) ---------------------------------------------------
    def read_first4(p):
        with open(p, "rb") as fh: return fh.read(4)

    hdr4 = read_first4(in_path)
    header_says_wav = hdr4 in (b"RIFF", b"RF64")
    if args.force_wav:
        is_wav = True
    elif args.force_raw:
        is_wav = False
    elif args.sr and args.sr > 0 and not header_says_wav:
        is_wav = False  # Only treat as RAW if sr provided AND not a WAV/RF64 file
    else:
        is_wav = header_says_wav

    if is_wav:
        try:
            # For large files, get header info first without loading entire file
            import soundfile as sf
            with sf.SoundFile(str(in_path)) as f:
                wav_sr = f.samplerate
                total_T = len(f)
            sr = float(args.sr) if args.sr > 0 else float(wav_sr)
            total_dur = total_T / sr

            # Only load full file if it's small, otherwise use chunk-based processing
            file_size_gb = os.path.getsize(in_path) / (1024**3)
            if file_size_gb < 0.5:  # < 500MB, load everything
                x_all, _ = load_wav_iq(str(in_path), channel_order=args.channel_order)
                print(f"[wav] {in_path.name} loaded into memory. sr={sr} Hz; duration={total_dur:.3f}s")
            else:
                x_all = None  # Will load chunks on demand
                print(f"[wav] {in_path.name} large file ({file_size_gb:.1f}GB), using streaming mode. sr={sr} Hz; duration={total_dur:.3f}s")
        except Exception as e:
            print(f"[warn] WAV parse failed for {in_path.name} ({e}). Falling back to RAW IQ.")
            is_wav = False

    if not is_wav:
        if args.sr <= 0:
            raise ValueError(f"RAW mode requires --sr (Hz); file: {in_path}")
        sr = float(args.sr)
        bytes_total = os.path.getsize(in_path)
        total_samples = bytes_total // BYTES_PER_COMPLEX
        total_T = int(total_samples)
        total_dur = total_T / sr
        print(f"[raw] {in_path.name} sr={sr:.0f} Hz; duration={total_dur:.3f}s; total_samples={total_samples:,}")

    # ---- Chunk plan -----------------------------------------------------------------
    seg_samples   = int(round(sr * args.seg))
    ovl_samples   = int(round(sr * args.overlap))
    if ovl_samples >= seg_samples:
        ovl_samples = 0
    hop_samples   = seg_samples - ovl_samples
    start_samples = int(round(sr * args.start))
    start_samples = max(0, start_samples)
    if start_samples >= total_T:
        raise ValueError(f"Start offset beyond end of file for {in_path.name}")

    remaining = total_T - start_samples
    n_chunks  = int(math.ceil(max(0, (remaining - seg_samples)) / float(hop_samples))) + 1
    if args.limit > 0:
        n_chunks = min(n_chunks, args.limit)

    print(f"[plan] {in_path.name} chunks={n_chunks} seg={args.seg:.3f}s overlap={args.overlap:.3f}s start={args.start:.3f}s")

    # ---- Optional recording start time parsing --------------------------------------
    rec_start_dt = None
    if args.recording_start:
        try:
            s = args.recording_start
            if s.endswith('Z'):
                rec_start_dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
            else:
                rec_start_dt = datetime.fromisoformat(s)
        except Exception:
            print("[warn] Could not parse --recording_start; expected ISO8601. Ignoring.")
            rec_start_dt = None

    # ---- Pass 1: split write (if requested) -----------------------------------------
    if args.split_write:
        print(f"[split] Writing chunk files for {in_path.name} …")
        pbar = tqdm(total=n_chunks, desc=f"Split {in_path.stem}", unit="chunk")
        for idx in range(n_chunks):
            s0 = start_samples + idx * hop_samples
            s1 = min(total_T, s0 + seg_samples)
            if s0 >= s1:
                break
            if is_wav:
                x_chunk = x_all[:, s0:s1]
            else:
                x_chunk = load_r64(str(in_path), channel_order=args.channel_order, start=s0, end=s1)
            # Write as .r64 (float32 interleaved)
            out_r64 = chunks_dir / f"chunk_{idx:04d}.r64"
            interleaved = np.empty(x_chunk.shape[1] * 2, dtype=np.float32)
            interleaved[0::2] = x_chunk[0]
            interleaved[1::2] = x_chunk[1]
            interleaved.tofile(out_r64)
            pbar.update(1)
        pbar.close()

    # ---- Pass 2: inference over chunks ----------------------------------------------
    print(f"[chunks] Starting inference on {n_chunks} chunks...")
    merged_rows = []
    frame_csv_paths = []
    pbar = tqdm(total=n_chunks, desc=f"Infer {in_path.stem}", unit="chunk")
    for idx in range(n_chunks):
        print(f"[chunk-{idx}] Loading chunk data...")
        s0 = start_samples + idx * hop_samples
        s1 = min(total_T, s0 + seg_samples)
        if s0 >= s1:
            break
        if args.split_write:
            chpath = chunks_dir / f"chunk_{idx:04d}.r64"
            x_chunk = load_r64(chpath, channel_order="IQ", start=0, end=None)  # written as IQ
        else:
            if is_wav:
                if x_all is not None:
                    # Small file loaded in memory
                    x_chunk = x_all[:, s0:s1]
                else:
                    # Large file streaming mode - load chunk on demand
                    print(f"[chunk-{idx}] Using streaming read for samples {s0}:{s1} ({(s1-s0)/sr:.1f}s)")
                    import soundfile as sf
                    try:
                        with sf.SoundFile(str(in_path)) as f:
                            f.seek(s0)
                            samples_to_read = s1 - s0
                            print(f"[chunk-{idx}] Reading {samples_to_read:,} samples from file...")
                            data = f.read(samples_to_read, always_2d=True, dtype='float32').T
                            print(f"[chunk-{idx}] Read data shape: {data.shape}")
                            if data.shape[0] == 1:
                                # Mono - assume interleaved I/Q
                                mono = data[0]
                                I = mono[0::2]
                                Q = mono[1::2]
                                x_chunk = np.stack([I, Q], axis=0)
                            else:
                                # Stereo - use first two channels
                                x_chunk = data[:2]
                            if args.channel_order.upper() == "QI":
                                x_chunk = x_chunk[[1, 0]]
                            print(f"[chunk-{idx}] Processed IQ data shape: {x_chunk.shape}")
                    except Exception as e:
                        print(f"[chunk-{idx}] Streaming read failed: {e}")
                        raise
            else:
                x_chunk = load_r64(str(in_path), channel_order=args.channel_order, start=s0, end=s1)

        print(f"[chunk-{idx}] Data loaded, starting classification...")
        rows, frame_csv = classify_segment(
            x_chunk, sr, labels, model, device, args, run_dir,
            seg_index=idx, global_offset_s=s0 / float(sr),
            save_frames=args.save_frames, save_spectrograms=args.save_spectrograms, amp_on=args.amp
        )
        print(f"[chunk-{idx}] Classification completed, {len(rows)} segments found")
        merged_rows.extend(rows)
        if frame_csv:
            frame_csv_paths.append(str(frame_csv))
        pbar.update(1)
        if getattr(args, 'cuda_gc', False) and device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    pbar.close()

    # ---- Group into bursts and annotate absolute fields -----------------------------
    bursts = group_bursts(merged_rows, merge_gap_s=args.group_gap_ms / 1000.0)
    for b in bursts:
        b["file"] = in_path.name  # annotate file for consolidation
        if args.rf_center_hz is not None:
            b["rf_center_hz_abs"] = round(args.rf_center_hz + b.get("rf_offset_hz", 0.0), 1)
        else:
            b["rf_center_hz_abs"] = ''
        if rec_start_dt is not None:
            t0 = rec_start_dt + timedelta(seconds=b["start_time_s"])
            t1 = rec_start_dt + timedelta(seconds=b["end_time_s"])
            b["start_time_abs"] = t0.isoformat()
            b["end_time_abs"] = t1.isoformat()
        else:
            b["start_time_abs"] = ''
            b["end_time_abs"] = ''

    # ---- Write per-file master CSV ---------------------------------------------------
    pred_csv = run_dir / "predictions.csv"
    header = [
        "burst_id", "pred_label", "avg_prob_top1",
        "start_time_s", "end_time_s", "duration_s", "start_time_abs", "end_time_abs",
        "rf_offset_hz", "rf_center_hz_abs", "bandwidth_hz", "snr_db"
    ]
    with open(pred_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for b in bursts:
            w.writerow({k: b.get(k, '') for k in header})

    # ---- Optional: merge frames ------------------------------------------------------
    if frame_csv_paths:
        frames_merged = run_dir / "frames.csv"
        with open(frames_merged, "w", newline="") as fout:
            w = None
            for ff in sorted(frame_csv_paths):
                with open(ff, "r", newline="") as fin:
                    r = csv.DictReader(fin)
                    if w is None:
                        w = csv.DictWriter(fout, fieldnames=r.fieldnames)
                        w.writeheader()
                    for row in r:
                        w.writerow(row)

    # ---- Optional timeline quicklook -------------------------------------------------
    if bursts:
        uniq = list(dict.fromkeys([b["pred_label"] for b in bursts]))
        base_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        colors = list(itertools.islice(itertools.cycle(base_colors), len(uniq)))
        label2c = {lab: colors[i] for i, lab in enumerate(uniq)}
        plt.figure(figsize=(14, 2.5))
        handles = {}
        for b in bursts:
            c = label2c[b["pred_label"]]
            h = plt.plot([b["start_time_s"], b["end_time_s"]], [1, 1], linewidth=8, color=c, label=b["pred_label"])[0]
            if b["pred_label"] not in handles:
                handles[b["pred_label"]] = h
        plt.yticks([])
        plt.xlabel("Time (s)")
        plt.title(f"Bursts timeline — {in_path.name}")
        plt.legend(handles.values(), handles.keys(), loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(run_dir / "timeline.png", bbox_inches="tight", dpi=160)
        plt.close()

    # ---- Write per-file run info -----------------------------------------------------
    info = {
        "source": os.path.abspath(str(in_path)),
        "sample_rate_hz": float(sr),
        "duration_s": float(total_T / sr),
        "chunks": int(n_chunks),
        "T_crop": int(args.T_crop),
        "stride": int(args.stride),
        "labels_count": int(len(labels)),
        "auto_vad": bool(args.auto_vad),
        "device": str(device),
        "amp": bool(args.amp),
        "out_dir": str(run_dir)
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"[OK] {in_path.name} -> {run_dir}")
    return bursts, run_dir, frame_csv_paths

# -------------- Main --------------

def main():
    ap = argparse.ArgumentParser("Wideband segmentation + classification with burst metrics")
    # Inputs / chunking
    ap.add_argument("--in", dest="in_path", required=True, help="Path to .wav / .r64 file OR a directory of such files")
    ap.add_argument("--channel_order", choices=["IQ","QI"], default="IQ")
    ap.add_argument("--sr", type=float, default=0.0, help="Sample rate Hz for raw/.r64 or raw-float32 WAV when using --force_raw")
    ap.add_argument("--force_raw", action="store_true", help="Treat input as raw float32 I/Q regardless of header")
    ap.add_argument("--force_wav", action="store_true", help="Treat input as WAV regardless of header")
    ap.add_argument("--seg", type=float, default=10.0, help="Chunk seconds (e.g., 5–10s for 20 MHz)")
    ap.add_argument("--overlap", type=float, default=0.0, help="Chunk overlap seconds")
    ap.add_argument("--start", type=float, default=0.0, help="Start offset seconds")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of chunks (0=no limit)")
    ap.add_argument("--split_write", action="store_true", help="Write chunk files to disk then read back for inference")

    # Model / labels
    ap.add_argument("--model_ts", required=True, help="TorchScript model (.pt)")
    ap.add_argument("--labels_json", required=True, help="JSON list[str] or dict with kept_class_names")
    ap.add_argument("--in_ch", type=int, default=14, help="Model input channels")

    # VAD / framing
    ap.add_argument("--T_crop", type=int, default=16384)
    ap.add_argument("--stride", type=int, default=8192)
    ap.add_argument("--vad_T", type=int, default=4096)
    ap.add_argument("--vad_stride", type=int, default=2048)
    ap.add_argument("--auto_vad", action="store_true")
    ap.add_argument("--auto_vad_quantile", type=float, default=0.98)
    ap.add_argument("--auto_vad_margin_db", type=float, default=6.0)
    ap.add_argument("--vad_thr_db", type=float, default=-35.0)
    ap.add_argument("--vad_hang_db", type=float, default=3.0)
    ap.add_argument("--min_dur_ms", type=float, default=60.0)
    ap.add_argument("--merge_gap_ms", type=float, default=60.0)
    ap.add_argument("--fallback_fullfile", action="store_true")
    ap.add_argument("--group_gap_ms", type=float, default=60.0, help="Gap to group adjacent segments into a burst")

    # Extra VAD controls
    ap.add_argument("--vad_abs_dbfs", type=float, default=None, help="Absolute VAD threshold (dBFS); overrides auto if set")
    ap.add_argument("--auto_vad_smooth_ms", type=float, default=0.54, help="Smoothing window for auto VAD (ms)")
    ap.add_argument("--envelope_vad", action="store_true", help="Use amplitude envelope instead of power for auto-VAD")

    # Fast auto-VAD
    ap.add_argument("--fast_vad", action="store_true", help="Speed up auto-VAD by downsampling the envelope for threshold estimation")
    ap.add_argument("--vad_down", type=int, default=100, help="Downsample factor for --fast_vad")
    ap.add_argument("--auto_vad_max_points", type=int, default=2000000, help="Cap points used for auto-VAD noise estimation")

    # Outputs / device / timing
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--out_dir", default="step5_out")
    ap.add_argument("--tag", default="burst20m")
    ap.add_argument("--save_spectrograms", action="store_true")
    ap.add_argument("--save_frames", action="store_true")
    ap.add_argument("--unknown_by_conf", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true")

    # OOM resilience
    ap.add_argument("--auto_batch", action="store_true", help="Dynamically shrink batch size on CUDA OOM")
    ap.add_argument("--min_batch", type=int, default=16, help="Minimum batch size when auto-batching")
    ap.add_argument("--cuda_gc", action="store_true", help="Empty CUDA cache between segments/chunks")

    # RF annotations
    ap.add_argument("--rf_center_hz", type=float, default=None, help="Absolute RF center freq (Hz) for absolute frequency column")
    ap.add_argument("--recording_start", type=str, default=None, help="Wallclock start time, e.g., 2025-09-19T10:00:00Z or 2025-09-19 10:00:00+08:00")

    # Retune + decimate
    ap.add_argument("--retune_decim", action="store_true", help="Frequency-shift each VAD segment to baseband and decimate before inference")
    ap.add_argument("--target_sr", type=float, default=1_000_000.0, help="Target sample rate after decimation (Hz)")
    ap.add_argument("--lp_taps", type=int, default=127, help="Low-pass FIR taps for decimation")
    ap.add_argument("--lp_pass", type=float, default=0.40, help="Passband as fraction of Nyquist at new SR")

    ap.add_argument("--psd_nfft", type=int, default=4096, help="FFT size for streaming PSD")
    ap.add_argument("--psd_hop", type=int, default=1024, help="Hop for streaming PSD")
    ap.add_argument("--psd_max_frames", type=int, default=8192, help="Max frames to average in PSD (caps work/memory)")
    ap.add_argument("--psd_preview_sr", type=float, default=2_000_000.0, help="Downsample to this SR for PSD only (None to disable)")


    args = ap.parse_args()

    # ---- Prepare output root for this run -------------------------------------------
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_root = out_root / f"{stamp}_{args.tag}"
    base_root.mkdir(parents=True, exist_ok=True)

    # ---- Load labels once ------------------------------------------------------------
    with open(args.labels_json, "r") as f:
        raw_labels = json.load(f)
    if isinstance(raw_labels, list) and all(isinstance(x, str) for x in raw_labels):
        labels = raw_labels
    elif isinstance(raw_labels, dict):
        labels = raw_labels.get("kept_class_names") or raw_labels.get("labels")
    else:
        labels = None
    if not labels:
        raise ValueError("labels_json must be a list[str] or dict with kept_class_names/labels")

    # ---- Device & model once ---------------------------------------------------------
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")
    try:
        torch.set_default_device(device.type)
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print(f"[model] Loading model from {args.model_ts}...")
    model = torch.jit.load(args.model_ts, map_location=device)
    print(f"[model] Model loaded, moving to device {device}...")
    try:
        model.to(device)
    except Exception:
        pass
    model.eval()
    print(f"[model] Optimizing model parameters...")
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (torch.nn.GRU, torch.nn.LSTM)):
                try:
                    m.flatten_parameters()
                except Exception:
                    pass
    print(f"[model] Model ready for inference")

    # Ensure labels count matches model output
    dummy = torch.zeros(1, args.in_ch, args.T_crop, dtype=torch.float32, device=device)
    out_dim = int(model(dummy).shape[-1])
    if len(labels) < out_dim:
        labels = labels + [f"class_{i}" for i in range(len(labels), out_dim)]
    elif len(labels) > out_dim:
        labels = labels[:out_dim]

    # ---- Build input list (file or directory) ---------------------------------------
    in_arg = Path(args.in_path)
    if in_arg.is_dir():
        # Non-recursive; change to rglob if you want recursion
        files = sorted(list(in_arg.glob("*.wav")) + list(in_arg.glob("*.r64")))
        if not files:
            raise FileNotFoundError(f"No .wav or .r64 files found in folder: {in_arg}")
    else:
        if not in_arg.exists():
            raise FileNotFoundError(f"Input path does not exist: {in_arg}")
        files = [in_arg]

    print(f"[batch] Found {len(files)} file(s). Output root: {base_root}")

    # ---- Run all files & consolidate ------------------------------------------------
    all_bursts = []
    for fp in files:
        bursts, file_run_dir, _ = run_on_file(str(fp), args, labels, model, device, base_root, stamp)
        all_bursts.extend(bursts)

    # Consolidated CSV across all files
    if all_bursts:
        consolidated_csv = base_root / "consolidated_predictions.csv"
        header = [
            "file",
            "burst_id", "pred_label", "avg_prob_top1",
            "start_time_s", "end_time_s", "duration_s",
            "start_time_abs", "end_time_abs",
            "rf_offset_hz", "rf_center_hz_abs", "bandwidth_hz", "snr_db"
        ]
        with open(consolidated_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for b in all_bursts:
                # write with file annotation
                row = {k: b.get(k, '') for k in header}
                w.writerow(row)
        print(f"[OK] Consolidated bursts -> {consolidated_csv}")
    else:
        print("[WARN] No bursts found across inputs.")

    # Final note
    print("[DONE] Batch run folder:", base_root)

if __name__ == "__main__":
    main()

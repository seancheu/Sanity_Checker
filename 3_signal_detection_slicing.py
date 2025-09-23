#!/usr/bin/env python3
"""
Step-3: Signal detection & slicing (STREAMING, RAM-safe)

- Scans a huge I/Q WAV in chunks (no full-file load)
- Detects active regions using short-term RMS threshold (dBFS)
- Merges adjacent activity with configurable padding
- Writes slice metadata (CSV + JSON) into the output run folder
- Optionally writes audio slices back to WAV files

Usage:
  python 3_signal_detection_slicing.py your.wav --out out_report\run_<...> \
    --fs_hint 20000000 --mem_budget_mb 512 --rms_win_ms 5 --thresh_dbfs -60 \
    --min_dur_ms 5 --gap_ms 3 --pad_ms 2 --write_audio

Notes:
- This step does *not* require IQ convention; it looks at amplitude only.
- Slices are saved under <out>/slices/ if --write_audio is given.
"""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

# ---------- helpers ----------

def dbfs(x_rms: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # x is amplitude in [-1,1]; 0 dBFS corresponds to full-scale sine ~ 0.707 rms (but we take 1.0 as 0 dBFS)
    return 20.0 * np.log10(np.maximum(x_rms, eps))

def running_rms_mag(block_mag: np.ndarray, win: int) -> np.ndarray:
    """
    Block-based RMS over magnitude envelope using cumulative sum of squares.
    Returns an array of length len(block_mag) with windowed RMS (centered via causal window).
    """
    if win <= 1:
        return block_mag.astype(np.float64, copy=False)
    x = block_mag.astype(np.float64, copy=False)
    cs = np.cumsum(x * x)
    out = np.empty_like(x)
    # y[n] = sqrt((cs[n] - cs[n-win])/win), handle beginning with smaller window
    for n in range(len(x)):
        a = 0 if n - win < 0 else cs[n - win]
        b = cs[n]
        w = n + 1 if n < win else win
        out[n] = math.sqrt((b - a) / max(1, w))
    return out

def merge_intervals(active: np.ndarray, fs: float, min_dur_ms: float, gap_ms: float, pad_ms: float) -> List[Tuple[int,int]]:
    """
    Convert boolean activity vector into merged [start,end) sample intervals
    - min_dur_ms: minimum active duration
    - gap_ms: merge if inactive gaps shorter than this
    - pad_ms: extend each interval on both sides
    """
    N = active.size
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return []

    # Find contiguous runs
    runs = []
    start = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
            continue
        runs.append((start, prev + 1))
        start = k
        prev = k
    runs.append((start, prev + 1))

    # Merge runs separated by small gaps
    merged = []
    gap_samps = int(round((gap_ms / 1000.0) * fs))
    cur_s, cur_e = runs[0]
    for s, e in runs[1:]:
        if s - cur_e <= gap_samps:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # Apply padding and duration filter
    pad_samps = int(round((pad_ms / 1000.0) * fs))
    min_samps = int(round((min_dur_ms / 1000.0) * fs))
    final = []
    for s, e in merged:
        s2 = max(0, s - pad_samps)
        e2 = min(N, e + pad_samps)
        if (e2 - s2) >= min_samps:
            final.append((s2, e2))
    return final

def compute_chunk_frames(mem_budget_mb: int, channels: int, floor: int = 1<<18) -> int:
    """Choose chunk frames so chunk*channels*4B <= mem_budget and >= floor."""
    mem_bytes = max(64, mem_budget_mb) * 1024 * 1024
    max_frames = mem_bytes // (channels * 4)
    if max_frames < floor:
        return floor
    k = max(1, max_frames // floor)
    return int(min(k * floor, 16 * 1024 * 1024))  # cap ~16M

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Step-3: streaming signal detection & slicing")
    ap.add_argument("wav", help="Path to I/Q .wav (stereo)")
    ap.add_argument("--out", default="out_report", help="Output folder (use the run dir from Step-1)")
    ap.add_argument("--fs_hint", type=float, default=None, help="Override WAV header Fs (Hz)")
    ap.add_argument("--mem_budget_mb", type=int, default=512, help="RAM budget per chunk (MB)")
    # Detection params
    ap.add_argument("--rms_win_ms", type=float, default=5.0, help="RMS window (ms)")
    ap.add_argument("--thresh_dbfs", type=float, default=-60.0, help="Activity threshold (dBFS)")
    ap.add_argument("--min_dur_ms", type=float, default=5.0, help="Minimum slice duration (ms)")
    ap.add_argument("--gap_ms", type=float, default=3.0, help="Merge gaps shorter than this (ms)")
    ap.add_argument("--pad_ms", type=float, default=2.0, help="Pad each slice on both sides (ms)")
    # Output control
    ap.add_argument("--write_audio", action="store_true", help="Write per-slice WAV files")
    ap.add_argument("--max_slices", type=int, default=1000, help="Safety cap for number of slices")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slices_dir = out_dir / "slices"
    if args.write_audio:
        slices_dir.mkdir(parents=True, exist_ok=True)

    info = sf.info(args.wav)
    fs_hdr = float(info.samplerate)
    fs = float(args.fs_hint) if args.fs_hint else fs_hdr
    C = int(info.channels)
    N_total = int(info.frames)
    if C < 2:
        raise SystemExit(f"Expected stereo I/Q, got {C} channel(s).")

    print(f"[INFO] WAV: {args.wav}")
    print(f"[INFO] Fs: {fs/1e6:.3f} MHz  Channels: {C}  Frames: {N_total:,}  Duration: {N_total/fs:.2f}s")

    chunk_frames = compute_chunk_frames(args.mem_budget_mb, channels=C, floor=1<<18)
    print(f"[INFO] Streaming in chunks of {chunk_frames:,} frames (~{chunk_frames/fs:.3f}s/chunk)")

    # First pass: build activity vector (True/False) over entire file at sample rate.
    # To avoid allocating N_total booleans for huge files, we collect intervals per chunk and map to global.
    intervals_global: List[Tuple[int, int]] = []

    win_samps = max(1, int(round(fs * (args.rms_win_ms / 1000.0))))
    thresh = float(args.thresh_dbfs)

    start_frame = 0
    with sf.SoundFile(args.wav, "r") as f:
        while start_frame < N_total:
            f.seek(start_frame)
            to_read = min(chunk_frames, N_total - start_frame)
            block = f.read(to_read, always_2d=True, dtype="float32")
            if block.size == 0:
                break

            # Magnitude envelope from I/Q
            I = block[:, 0]
            Q = block[:, 1]
            mag = np.sqrt(I.astype(np.float64)**2 + Q.astype(np.float64)**2)

            # Short-term RMS (causal window), convert to dBFS
            rms = running_rms_mag(mag, win_samps)
            db = dbfs(rms)

            # Active where above threshold
            active = db > thresh

            # Merge within chunk → local intervals
            local = merge_intervals(active, fs, args.min_dur_ms, args.gap_ms, args.pad_ms)

            # Map local intervals to global sample indices
            for s, e in local:
                gs = s + start_frame
                ge = e + start_frame
                intervals_global.append((gs, ge))

            start_frame += to_read

    # Second pass: merge global intervals across chunk boundaries
    if not intervals_global:
        print("[INFO] No activity detected above threshold.")
        intervals_merged = []
    else:
        intervals_global.sort()
        merged = []
        cur_s, cur_e = intervals_global[0]
        for s, e in intervals_global[1:]:
            if s <= cur_e:  # overlap/adjacent
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        intervals_merged = merged

    # Cap number of slices
    if len(intervals_merged) > args.max_slices:
        print(f"[WARN] Detected {len(intervals_merged)} slices; capping to first {args.max_slices}.")
        intervals_merged = intervals_merged[:args.max_slices]

    # Write metadata
    meta = {
        "wav": str(Path(args.wav).resolve()),
        "fs_hz": fs,
        "frames": N_total,
        "duration_s": N_total / fs,
        "params": {
            "mem_budget_mb": args.mem_budget_mb,
            "rms_win_ms": args.rms_win_ms,
            "thresh_dbfs": args.thresh_dbfs,
            "min_dur_ms": args.min_dur_ms,
            "gap_ms": args.gap_ms,
            "pad_ms": args.pad_ms,
            "chunk_frames": chunk_frames
        },
        "num_slices": len(intervals_merged)
    }
    (out_dir / "slices_meta.json").write_text(json.dumps(meta, indent=2))

    with open(out_dir / "slices.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["slice_idx", "start_sample", "end_sample", "start_s", "end_s", "duration_ms"])
        for i, (s, e) in enumerate(intervals_merged):
            w.writerow([i, s, e, s / fs, e / fs, (e - s) * 1000.0 / fs])

    print(f"[INFO] Slices found: {len(intervals_merged)} (metadata written to {out_dir/'slices.csv'})")

    # Optional: write audio slices as .wav (still streaming)
    if args.write_audio and intervals_merged:
        with sf.SoundFile(args.wav, "r") as f:
            for i, (s, e) in enumerate(intervals_merged):
                f.seek(s)
                frames = e - s
                # stream write in sub-chunks to avoid big allocations
                out_path = slices_dir / f"slice_{i:05d}.wav"
                with sf.SoundFile(out_path, "w", samplerate=int(fs), channels=C, subtype="PCM_16") as out_f:
                    remaining = frames
                    while remaining > 0:
                        n = min(262144, remaining)
                        blk = f.read(n, always_2d=True, dtype="float32")
                        if blk.size == 0:
                            break
                        out_f.write(blk)
                        remaining -= blk.shape[0]
        print(f"[INFO] Wrote {len(intervals_merged)} WAV slices to {slices_dir}")

    print(f"[DONE] Step-3 completed in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()

"""
A) Slice the whole file per carrier (good starting point)
python slice_signals.py your_20MHz.wav
  --fs_hint 20000000
  --carriers_csv out_report/carriers.csv
  --mode carriers
  --win 16384 --hop_frac 0.5
  --oversample 4.0 --min_bw 100000
  --out slices_out

B) Slice only when active (better for bursty signals)
python slice_signals.py your_20MHz.wav
  --fs_hint 20000000
  --carriers_csv out_report/carriers.csv
  --mode bursts
  --nperseg 4096 --overlap 0.5 --cfar_k 3.0
  --win 16384 --hop_frac 0.5
  --oversample 4.0 --min_bw 100000
  --out slices_out_bursty
"""

# slice_signals.py — Step 3: down-mix, low-pass, decimate, and window IQ per detected signal

import argparse, json, csv
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal as sig
import h5py
import matplotlib.pyplot as plt

# ---------- Utilities shared with Step 2 ----------
def to_complex_iq(x, conv="I+Q"):
    I, Q = x[:,0].astype(np.float32), x[:,1].astype(np.float32)
    if conv == "I+Q":   return I + 1j*Q
    if conv == "I-Q":   return I - 1j*Q
    if conv == "Q+I":   return Q + 1j*I
    raise ValueError(f"Unknown conv '{conv}'")

def dc_remove(xc): return xc - np.mean(xc)

def welch_psd(xc, fs, nperseg, noverlap):
    f, Pxx = sig.welch(
        xc, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
        return_onesided=False, detrend=False, scaling="density"
    )
    f = np.fft.fftshift(f); Pxx = np.fft.fftshift(Pxx)
    return f, Pxx

def find_carriers(f, psd_db, floor_db, min_prom_db=8.0, min_bw_bins=3):
    peaks, props = sig.find_peaks(psd_db, prominence=min_prom_db)
    carriers = []
    if len(f) < 2: return carriers
    df = float(f[1] - f[0])
    for p in peaks:
        pk_db = float(psd_db[p])
        thr = pk_db - 6.0  # -6 dB width
        l = p
        while l > 0 and psd_db[l] > thr: l -= 1
        r = p
        while r < len(psd_db)-1 and psd_db[r] > thr: r += 1
        if (r - l) < min_bw_bins: continue
        carriers.append({
            "f_center_hz": float(f[p]),
            "p_db": pk_db,
            "bw_hz_est": float((r - l) * df)
        })
    return carriers

def stft_waterfall(xc, fs, nperseg, noverlap):
    f, t, Z = sig.stft(
        xc, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
        return_onesided=False, boundary=None, detrend=False
    )
    f = np.fft.fftshift(f); Z = np.fft.fftshift(Z, axes=0)
    S_db = 20*np.log10(np.maximum(np.abs(Z), 1e-12))
    return f, t, S_db

def cfar_mask(S_db, guard=1, train=6, k=3.0):
    F, T = S_db.shape
    mask_t = np.zeros_like(S_db, dtype=bool)
    for t in range(T):
        t0 = max(0, t - train - guard); t1 = max(0, t - guard)
        t2 = min(T, t + guard + 1);     t3 = min(T, t + guard + 1 + train)
        ref = np.concatenate([S_db[:, t0:t1], S_db[:, t2:t3]], axis=1)
        if ref.size == 0: continue
        mu = np.median(ref, axis=1, keepdims=True)
        mad = np.median(np.abs(ref - mu), axis=1, keepdims=True) + 1e-12
        thr = mu + 1.4826 * mad * k
        mask_t[:, t] = S_db[:, t:t+1] > thr
    mask_f = np.zeros_like(S_db, dtype=bool)
    for f_i in range(F):
        f0 = max(0, f_i - train - guard); f1 = max(0, f_i - guard)
        f2 = min(F, f_i + guard + 1);     f3 = min(F, f_i + guard + 1 + train)
        ref = np.concatenate([S_db[f0:f1, :], S_db[f2:f3, :]], axis=0)
        if ref.size == 0: continue
        mu = np.median(ref, axis=0, keepdims=True)
        mad = np.median(np.abs(ref - mu), axis=0, keepdims=True) + 1e-12
        thr = mu + 1.4826 * mad * k
        mask_f[f_i, :] = S_db[f_i:f_i+1, :] > thr
    return mask_t & mask_f

# ---------- Core slicing helpers ----------
def design_lp(fs, cutoff_hz, trans=0.15, taps=801):
    # FIR low-pass with Hann; cutoff is passband edge; simple & robust
    nyq = fs * 0.5
    wc = min(0.99, float(cutoff_hz / nyq))
    if wc <= 0: wc = 0.01
    return sig.firwin(taps, wc, window="hann")

def downmix_decimate(xc, fs, f0, bw_est, oversample=4.0, min_bw=1e5, taps=801):
    # Down-mix to baseband, low-pass to ~1.2×BW, and decimate to >= 4×BW
    bw = max(float(bw_est), float(min_bw))
    decim = max(1, int(np.floor(fs / (bw * oversample))))
    # Mix
    n = np.arange(xc.shape[0], dtype=np.float64)
    mixer = np.exp(-1j * 2.0 * np.pi * (f0 / fs) * n)
    y = xc * mixer.astype(np.complex64)
    # LPF ~ 1.2× BW
    cutoff = min(0.45 * fs, 1.2 * bw)
    b = design_lp(fs, cutoff, taps=taps)
    y = sig.lfilter(b, [1.0], y)
    # Decimate (polyphase)
    y_dec = sig.resample_poly(y, up=1, down=decim, window=('kaiser', 8.6))
    fs_dec = fs / decim
    return y_dec.astype(np.complex64, copy=False), fs_dec, decim, cutoff

def slice_windows(yc, win, hop):
    N = yc.shape[0]
    if N < win: return []
    starts = np.arange(0, N - win + 1, hop, dtype=int)
    return [(s, s+win) for s in starts]

def time_segments_from_mask(f_axis, t_axis, mask, f0, bw, freq_pad=1.25):
    # Reduce 2D mask to 1D "active over time" near [f0 ± pad*(bw/2)]
    f_lo = f0 - 0.5 * bw * freq_pad
    f_hi = f0 + 0.5 * bw * freq_pad
    band = (f_axis >= f_lo) & (f_axis <= f_hi)
    if not np.any(band):
        return []
    active_t = mask[band, :].any(axis=0)  # [T]
    # group contiguous true runs
    segs = []
    in_run = False
    t0 = 0
    for i, a in enumerate(active_t):
        if a and not in_run:
            in_run = True; t0 = i
        if not a and in_run:
            in_run = False; segs.append((t0, i-1))
    if in_run: segs.append((t0, len(active_t)-1))
    # Convert STFT frame indices to time seconds
    out = []
    for s0, s1 in segs:
        t_start = float(t_axis[max(0, s0)])
        t_end   = float(t_axis[min(len(t_axis)-1, s1)])
        out.append((t_start, t_end))
    return out

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="Path to I/Q WAV")
    ap.add_argument("--out", default="slices_out", help="Output folder")
    ap.add_argument("--fs_hint", type=float, default=None, help="If WAV header is wrong")
    ap.add_argument("--conv", default="I+Q", choices=["I+Q","I-Q","Q+I"], help="I/Q convention")
    ap.add_argument("--carriers_csv", default=None, help="Use carriers from Step-2 CSV; else re-detect")
    ap.add_argument("--prom_db", type=float, default=8.0, help="Min PSD prominence for re-detect")
    ap.add_argument("--nperseg", type=int, default=4096, help="STFT/PSD window")
    ap.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction")
    ap.add_argument("--mode", choices=["carriers","bursts"], default="carriers",
                    help="'carriers' = slice whole file per carrier; 'bursts' = slice only when active")
    ap.add_argument("--cfar_k", type=float, default=3.0)
    ap.add_argument("--cfar_guard", type=int, default=1)
    ap.add_argument("--cfar_train", type=int, default=6)
    ap.add_argument("--win", type=int, default=16384, help="Clip length (samples, AFTER decimation)")
    ap.add_argument("--hop_frac", type=float, default=0.5, help="Window hop as fraction of win")
    ap.add_argument("--oversample", type=float, default=4.0, help="Fs' target ≈ oversample×BW")
    ap.add_argument("--min_bw", type=float, default=1e5, help="Minimum BW assumption (Hz)")
    ap.add_argument("--taps", type=int, default=801, help="FIR taps for low-pass before decimation")
    ap.add_argument("--qc_plots", action="store_true", help="Save per-carrier PSD and a few clip PSDs")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Load WAV
    x, fs_hdr = sf.read(args.wav, always_2d=True)
    x = x.astype(np.float32, copy=False)
    fs = float(args.fs_hint) if args.fs_hint else float(fs_hdr)
    N, C = x.shape
    if C < 2: raise SystemExit(f"Expected stereo I/Q, got {C} ch.")
    dur = N / fs

    # Complex IQ and DC removal
    xc = dc_remove(to_complex_iq(x, conv=args.conv))

    # Carrier list
    if args.carriers_csv and Path(args.carriers_csv).exists():
        carriers = []
        with open(args.carriers_csv, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                carriers.append({
                    "f_center_hz": float(row["f_center_hz"]),
                    "bw_hz_est": float(row["bw_hz_est"]),
                    "p_db": float(row.get("p_db", 0.0))
                })
    else:
        # Re-detect from PSD
        f_psd, Pxx = welch_psd(xc, fs, args.nperseg, int(args.nperseg*args.overlap))
        Pxx_db = 10*np.log10(np.maximum(Pxx, 1e-20))
        floor_db = float(np.median(Pxx_db))
        carriers = find_carriers(f_psd, Pxx_db, floor_db, min_prom_db=args.prom_db)

    if not carriers:
        print("No carriers found. Exiting.")
        return

    # Optional: STFT/CFAR for burst-gated slicing
    f_stft = t_stft = S_db = mask = None
    if args.mode == "bursts":
        f_stft, t_stft, S_db = stft_waterfall(xc, fs, args.nperseg, int(args.nperseg*args.overlap))
        mask = cfar_mask(S_db, guard=args.cfar_guard, train=args.cfar_train, k=args.cfar_k)

    # Prepare HDF5
    h5_path = Path(args.out, "slices.h5")
    meta = {"file": str(Path(args.wav).resolve()),
            "fs_used_hz": fs, "duration_s": float(dur),
            "clips": []}
    # Count total clips first? We'll append using resizeable dataset.
    dset = None

    for idx, c in enumerate(carriers):
        f0 = c["f_center_hz"]; bw_est = c["bw_hz_est"]
        y_dec, fs_dec, decim, cutoff = downmix_decimate(
            xc, fs, f0, bw_est, oversample=args.oversample, min_bw=args.min_bw, taps=args.taps
        )
        # Decide time spans to slice
        spans = []
        if args.mode == "carriers":
            spans = [(0.0, dur)]
        else:
            # derive active time spans near this carrier
            spans = time_segments_from_mask(f_stft, t_stft, mask, f0, bw_est, freq_pad=1.25)

        # Build slices
        win = int(args.win)
        hop = max(1, int(win * args.hop_frac))
        clips_here = 0

        # map original time to decimated sample indices
        for (t0, t1) in spans:
            # segment in original samples
            n0 = int(np.floor(t0 * fs))
            n1 = int(min(N, np.ceil(t1 * fs)))
            if n1 <= n0 + 4: continue
            # get the already downmixed+filtered+decimated stream for the whole file,
            # then convert time to decimated indices:
            d0 = int(np.floor(n0 / decim))
            d1 = int(np.floor(n1 / decim))
            sl = y_dec[d0:d1]
            if sl.size < win: continue
            # window it
            for s, e in slice_windows(sl, win, hop):
                seg = sl[s:e]
                # init HDF5 dataset if needed
                if dset is None:
                    dset = h5py.File(h5_path, "w")
                    maxshape = (None, 2, win)
                    X = dset.create_dataset("X", shape=(0, 2, win), maxshape=maxshape, dtype="float32", compression="gzip")
                    Y = dset.create_dataset("y", shape=(0,), maxshape=(None,), dtype="int32")  # placeholder labels
                    M = dset
                else:
                    X = dset["X"]; Y = dset["y"]
                # append
                i_new = X.shape[0]
                X.resize((i_new+1, 2, win))
                Y.resize((i_new+1,))
                # complex -> [2,T]
                X[i_new] = np.stack([np.real(seg), np.imag(seg)], axis=0).astype(np.float32)
                Y[i_new] = 0  # unknown label for now

                # clip meta
                clip_meta = {
                    "clip_index": int(i_new),
                    "carrier_index": int(idx),
                    "f_center_hz_rel": float(f0),
                    "bw_hz_est": float(bw_est),
                    "fs_dec_hz": float(fs_dec),
                    "decim": int(decim),
                    "t0_s": float(t0),
                    "t1_s": float(t1),
                    "source_span_dec": [int(s)+d0, int(e)+d0]
                }
                meta["clips"].append(clip_meta)
                clips_here += 1

        # optional QC plots
        if args.qc_plots:
            # per-carrier PSD at decimated rate
            f_dec, P_dec = welch_psd(y_dec, fs_dec, min(4096, max(1024, win//8)), min(2048, max(512, win//16)))
            P_dec_db = 10*np.log10(np.maximum(P_dec, 1e-20))
            fig = plt.figure(figsize=(8,3))
            plt.plot(f_dec/1e3, P_dec_db)
            plt.title(f"Carrier {idx}: f0={f0/1e6:.3f} MHz, Fs'={fs_dec/1e6:.3f} MHz")
            plt.xlabel("Freq (kHz)"); plt.ylabel("dB/Hz")
            fig.savefig(Path(args.out, f"carrier_{idx:02d}_dec_psd.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

        print(f"Carrier {idx}: f0={f0/1e6:.3f} MHz, bw≈{bw_est/1e3:.1f} kHz → Fs'={fs_dec/1e6:.3f} MHz, clips={clips_here}")

    # Write meta
    Path(args.out, "meta.json").write_text(json.dumps(meta, indent=2))
    if dset is not None:
        dset.close()
        print(f"Saved HDF5 clips → {h5_path}")
    else:
        print("No clips written (windows too short or no active spans).")

if __name__ == "__main__":
    main()

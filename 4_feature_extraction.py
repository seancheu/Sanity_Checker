"""
A) Just features (CSV): (use this if using --extra_ch)
python features_and_channels.py
  --slices_h5 slices_out/slices.h5
  --meta_json slices_out/meta.json
  --out_dir step4_out

B) Features + engineered channels HDF5 (for your TCN-BiLSTM): (use this if using --extra_ch none)
python features_and_channels.py
  --slices_h5 slices_out/slices.h5
  --meta_json slices_out/meta.json
  --out_dir step4_out
  --emit_engineered_h5
  --extras "amp,phase,dfreq,cosphi,sinphi,d_amp,d2phase,cum40,cum41,cum42"
"""

# features_and_channels.py — Step 4: extract per-clip features (CSV) and build engineered-channel HDF5
import argparse, json
from pathlib import Path
import numpy as np
import h5py
import scipy.signal as sig
import csv

# ---------- low-level helpers ----------
def welch_psd(xc, fs, nperseg=2048, overlap=0.5):
    nperseg = int(nperseg); noverlap = int(nperseg*overlap)
    f, P = sig.welch(xc, fs=fs, window="hann", nperseg=nperseg,
                     noverlap=noverlap, return_onesided=False,
                     detrend=False, scaling="density")
    f = np.fft.fftshift(f); P = np.fft.fftshift(P)
    return f, P

def snr_and_bw(f, P, peak_prom_db=6.0):
    """Return (snr_db, bw_hz_at_minus6dB)."""
    Pdb = 10*np.log10(np.maximum(P, 1e-20))
    floor_db = float(np.median(Pdb))
    p = np.argmax(Pdb)
    pk_db = float(Pdb[p])
    # -6 dB width
    thr = pk_db - 6.0
    l = p
    while l > 0 and Pdb[l] > thr: l -= 1
    r = p
    while r < len(Pdb)-1 and Pdb[r] > thr: r += 1
    bw_hz = float((r - l) * (f[1]-f[0])) if len(f) > 1 else 0.0
    snr_db = pk_db - floor_db
    return snr_db, abs(bw_hz)

def spectral_flatness(P):
    """Geometric mean / arithmetic mean of power spectrum."""
    P = np.maximum(P, 1e-20)
    g = np.exp(np.mean(np.log(P)))
    a = np.mean(P)
    return float(g/a)

def envelope_kurtosis(xc):
    amp = np.abs(xc).astype(np.float64)
    mu = amp.mean(); sd = amp.std() + 1e-12
    z = (amp - mu)/sd
    return float(np.mean(z**4))

def cumulants_20_40_41_42(xc):
    """Fourth-order cumulants; classic definitions for complex baseband."""
    x = xc.astype(np.complex64)
    x = x - x.mean()
    P = (np.abs(x)**2).mean() + 1e-12
    m2 = (x**2).mean()
    m4 = (x**4).mean()
    Rxx = (x*np.conj(x)).mean()
    C20 = m2 / P
    C40 = (m4 - 3*(m2**2)) / (P**2)
    C41 = (((x**3)*np.conj(x)).mean() - 3*m2*Rxx) / (P**2)
    C42 = (m4 - np.abs(m2)**2 - 2*(Rxx**2)) / (P**2)
    # return real & imag parts explicitly for ML-friendliness
    def r(v): return float(np.real(v))
    def i(v): return float(np.imag(v))
    return r(C20), i(C20), r(C40), i(C40), r(C41), i(C41), r(C42), i(C42)

def cfo_estimate(xc, fs):
    """Coarse CFO: arg of 1-sample complex product averaged."""
    x = xc.astype(np.complex64)
    phi = np.angle(np.vdot(x[:-1], x[1:]))  # mean angle of x[n+1]*conj(x[n])
    freq = phi/(2*np.pi) * fs
    return float(freq)

# engineered channels
def engineered_channels(i, q, which):
    which = [w.strip().lower() for w in which.split(",") if w.strip()] if which else []
    I = i.astype(np.float32); Q = q.astype(np.float32)
    amp = np.sqrt(I*I + Q*Q).astype(np.float32)
    phase = np.unwrap(np.arctan2(Q, I)).astype(np.float32)
    dfreq = np.diff(phase, prepend=phase[0]).astype(np.float32)
    out = [("I", I), ("Q", Q)]
    base = {
        "amp": amp,
        "phase": phase,
        "dfreq": dfreq,
        "cosphi": np.cos(phase).astype(np.float32),
        "sinphi": np.sin(phase).astype(np.float32),
        "d_amp": np.diff(amp, prepend=amp[0]).astype(np.float32),
        "d2phase": np.diff(phase, n=2, prepend=phase[0], append=phase[-1]).astype(np.float32),
    }
    for k in which:
        if k in base:
            out.append((k, base[k]))
    # higher-order cumulants as broadcast channels if requested
    # (constant over time; occasionally useful)
    if "cum20" in which or "cum40" in which or "cum41" in which or "cum42" in which:
        x = (I + 1j*Q).astype(np.complex64)
        c20r,c20i,c40r,c40i,c41r,c41i,c42r,c42i = cumulants_20_40_41_42(x)
        T = I.shape[0]
        def bc(v): return np.full(T, v, dtype=np.float32)
        if "cum20" in which: out.append(("cum20", bc(c20r)))
        if "cum40" in which: out.append(("cum40", bc(c40r)))
        if "cum41" in which: out.append(("cum41", bc(c41r)))
        if "cum42" in which: out.append(("cum42", bc(c42r)))
    # simple band energies if requested (coarse)
    if "bande_lo" in which or "bande_hi" in which or "spec_flat" in which:
        # tiny STFT to get average low/high band energy
        win = min(256, max(128, (len(I)//64)//2*2))
        hop = win//2
        if win >= 64:
            hann = 0.5 - 0.5*np.cos(2*np.pi*np.arange(win)/win)
            # complex window
            frames = []
            x_c = (I + 1j*Q).astype(np.complex64)
            for s in range(0, len(I)-win+1, hop):
                frames.append(x_c[s:s+win]*hann)
            if frames:
                S = np.fft.rfft(np.stack(frames,0), axis=1)
                Pm = (np.abs(S)**2).mean(axis=0)
                nfft = len(Pm)
                lo_end = nfft//4
                loE = float(np.log(Pm[:lo_end].mean()+1e-12))
                hiE = float(np.log(Pm[lo_end:].mean()+1e-12))
                flat = float(np.exp(np.mean(np.log(Pm+1e-12)))/(Pm.mean()+1e-12))
                T = I.shape[0]
                if "bande_lo" in which: out.append(("bandE_lo", np.full(T, loE, np.float32)))
                if "bande_hi" in which: out.append(("bandE_hi", np.full(T, hiE, np.float32)))
                if "spec_flat" in which: out.append(("spec_flat", np.full(T, flat, np.float32)))
    names = [n for n,_ in out]
    arr = np.stack([a for _,a in out], axis=0).astype(np.float32)  # [C,T]
    return names, arr

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices_h5", required=True, help="Path to Step-3 slices.h5")
    ap.add_argument("--meta_json", required=True, help="Path to Step-3 meta.json")
    ap.add_argument("--out_dir", default="step4_out")
    ap.add_argument("--fs_key", default=None, help="Force a single Fs' (Hz) for all clips; else read per-clip from meta.json")
    ap.add_argument("--nperseg", type=int, default=2048)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--extras", default="amp,phase,dfreq,cosphi,sinphi,d_amp,d2phase",
                    help="Engineered channels to append after I,Q. Use names separated by comma.")
    ap.add_argument("--emit_engineered_h5", action="store_true",
                    help="Also write an HDF5 with engineered channels for DL models.")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # load meta
    meta = json.loads(Path(args.meta_json).read_text())
    fs_per_clip = {}
    if args.fs_key:
        fs_fixed = float(args.fs_key)
    else:
        for clip in meta.get("clips", []):
            fs_per_clip[int(clip["clip_index"])] = float(clip["fs_dec_hz"])

    # open slices
    with h5py.File(args.slices_h5, "r") as h5:
        X = h5["X"]              # [N, 2, T]
        Y = h5["y"]              # placeholder labels
        N, _, T = X.shape

        # prepare outputs
        feats_csv = out / "features.csv"
        fh = open(feats_csv, "w", newline="")
        w = csv.writer(fh)
        header = [
            "clip_index","fs_hz","T","snr_db","bw_hz","sfm","env_kurt",
            "c20_r","c20_i","c40_r","c40_i","c41_r","c41_i","c42_r","c42_i",
            "cfo_hz"
        ]
        w.writerow(header)

        # engineered h5 (optional)
        if args.emit_engineered_h5:
            eng_h5_path = out / "engineered.h5"
            eng_h5 = h5py.File(eng_h5_path, "w")
            # we'll discover channel count on first clip
            Xeng = None
            yeng = eng_h5.create_dataset("y", shape=(0,), maxshape=(None,), dtype="int32")
        else:
            eng_h5 = None

        for i in range(N):
            iq = X[i]  # [2,T]
            I = iq[0].astype(np.float32); Q = iq[1].astype(np.float32)
            xc = (I + 1j*Q).astype(np.complex64)

            if args.fs_key:
                fs = fs_fixed
            else:
                fs = fs_per_clip.get(i, None)
                if fs is None:
                    # fallback: assume meta has a global fs; else error
                    fs = float(meta.get("fs_used_hz", 1.0))

            # PSD features
            f, P = welch_psd(xc, fs, nperseg=args.nperseg, overlap=args.overlap)
            snr_db, bw_hz = snr_and_bw(f, P)
            sfm = spectral_flatness(P)
            kurt = envelope_kurtosis(xc)
            c20r,c20i,c40r,c40i,c41r,c41i,c42r,c42i = cumulants_20_40_41_42(xc)
            cfo = cfo_estimate(xc, fs)

            w.writerow([i, fs, T, snr_db, bw_hz, sfm, kurt,
                        c20r, c20i, c40r, c40i, c41r, c41i, c42r, c42i, cfo])

            # engineered channels
            if eng_h5 is not None:
                names, arr = engineered_channels(I, Q, args.extras)  # [C,T] with ['I','Q',...]
                if Xeng is None:
                    C = arr.shape[0]
                    Xeng = eng_h5.create_dataset("X", shape=(0, C, T), maxshape=(None, C, T),
                                                 dtype="float32", compression="gzip")
                    # store channel names for reference
                    eng_h5.attrs["channels"] = json.dumps(names)
                # append
                i_new = Xeng.shape[0]
                Xeng.resize((i_new+1, arr.shape[0], T))
                yeng.resize((i_new+1,))
                Xeng[i_new] = arr
                yeng[i_new] = 0  # unknown labels for now

        fh.close()
        if eng_h5 is not None:
            eng_h5.close()

    print(f"✅ Wrote features CSV → {feats_csv}")
    if args.emit_engineered_h5:
        print(f"✅ Wrote engineered channels HDF5 → {eng_h5_path}")

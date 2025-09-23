#!/usr/bin/env python3
"""
Main runner that orchestrates:
  1) 1_sanity_check_iq.py  (streaming, full-file)
  2) 2_wideband_exploration.py     (PSD, waterfall, CFAR, carriers)
  3) 3_signal_detection_slicing.py (streaming RMS-based detection & optional slicing)

- Keeps scripts separate.
- Ensures all write into the SAME per-run folder created by Step-1.
- Passes best I/Q convention from Step-1 into Step-2 automatically.
- ALWAYS runs Step-1 -> Step-2 -> Step-3.
- Streams each step's logs live (no more "hang" before output).

Usage:
  python run_iq_pipeline.py "C:\path\to\your.wav" --fs_hint 20000000
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# ---------- helpers ----------

def _which_python() -> str:
    # Use the current interpreter to avoid venv surprises
    return sys.executable or "python"

def _stream(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    """
    Run a command, stream stdout/stderr live to this console,
    and also collect the full combined text for parsing.
    Returns (returncode, combined_text).
    """
    # Merge stderr into stdout so ordering is preserved
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,          # line-buffered
        universal_newlines=True,
    )
    collected: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        collected.append(line)
        # print as-is without extra newline (line already has it)
        print(line, end="")
    proc.wait()
    return proc.returncode, "".join(collected)

def _extract_json(stdout: str) -> Optional[dict]:
    """
    Step-1 prints a big JSON object plus some lines.
    Grab the first {...} to the matching last } and parse.
    """
    try:
        first = stdout.find("{")
        last = stdout.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return None
        blob = stdout[first:last+1]
        return json.loads(blob)
    except Exception:
        return None

def _extract_run_dir(stdout: str, out_dir_base: Path, wav_stem: str) -> Optional[Path]:
    """
    Prefer parsing the explicit line:
      [INFO] Outputs written to: "out_report\\run_<stem>_YYYYMMDD_HHMMSS"
    Fallback: pick the newest 'run_<stem>_*' in out_dir_base.
    """
    m = re.search(r'Outputs written to:\s*"([^"]+)"', stdout)
    if m:
        p = Path(m.group(1))
        if p.exists():
            return p

    # Fallback: newest matching run folder
    candidates = sorted(
        (d for d in out_dir_base.glob(f"run_{wav_stem}_*") if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True
    )
    return candidates[0] if candidates else None

def _map_best_conv_to_step2(conv_step1: str) -> str:
    """
    Step-1 emits: 'I+jQ', 'I-jQ', 'Q+jI'
    Step-2 expects: 'I+Q',  'I-Q',  'Q+I'
    """
    mapping = {
        "I+jQ": "I+Q",
        "I-jQ": "I-Q",
        "Q+jI": "Q+I",
    }
    return mapping.get(conv_step1, "I+Q")

def _with_unbuffered(py: str, script: Path) -> List[str]:
    """
    Ensure child Python is unbuffered (-u) so logs flush immediately.
    """
    return [py, "-u", str(script.resolve())]

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Main runner for I/Q pipeline (Step-1 sanity -> Step-2 analysis -> Step-3 slicing).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("wav", help="Path to I/Q .wav (stereo)")
    ap.add_argument("--fs_hint", type=float, default=None, help="Override WAV header Fs (Hz)")
    ap.add_argument("--out_dir", default="out_report", help="Base output folder")

    # Step-1 options
    ap.add_argument("--sanity_script", default="1_sanity_check_iq.py", help="Path to Step-1 script")
    ap.add_argument("--mem_budget_mb", type=int, default=512, help="RAM budget per chunk for Step-1")
    ap.add_argument("--fft_size", type=int, default=None, help="FFT size per chunk for Step-1 (power of 2)")
    ap.add_argument("--sanity_path", default=None, help="Working dir for Step-1 (optional)")

    # Step-2 options
    ap.add_argument("--step2_script", default="2_wideband_exploration.py", help="Path to Step-2 script")
    ap.add_argument("--nperseg", type=int, default=4096, help="FFT window length")
    ap.add_argument("--overlap", type=float, default=0.5, help="STFT/Welch overlap fraction")
    ap.add_argument("--prom_db", type=float, default=8.0, help="Min prominence (dB) for PSD peaks")
    ap.add_argument("--cfar_k", type=float, default=3.0, help="CFAR k (higher=stricter)")
    ap.add_argument("--cfar_guard", type=int, default=1, help="CFAR guard cells")
    ap.add_argument("--cfar_train", type=int, default=6, help="CFAR training cells")
    ap.add_argument("--max_duration", type=float, default=60.0, help="Max processing duration for Step-2 sampling")
    ap.add_argument("--force_full", action="store_true", help="Force full Step-2 processing (no sampling)")
    ap.add_argument("--step2_path", default=None, help="Working dir for Step-2 (optional)")

    # Step-3 options (always run)
    ap.add_argument("--step3_script", default="3_signal_detection_slicing.py", help="Path to Step-3 script")
    ap.add_argument("--step3_mem_budget_mb", type=int, default=512, help="RAM budget per chunk for Step-3")
    ap.add_argument("--step3_rms_win_ms", type=float, default=5.0, help="RMS window (ms) for Step-3")
    ap.add_argument("--step3_thresh_dbfs", type=float, default=-60.0, help="Threshold (dBFS) for Step-3")
    ap.add_argument("--step3_min_dur_ms", type=float, default=5.0, help="Min duration (ms) for Step-3 slices")
    ap.add_argument("--step3_gap_ms", type=float, default=3.0, help="Merge gaps shorter than this (ms)")
    ap.add_argument("--step3_pad_ms", type=float, default=2.0, help="Padding (ms) for Step-3 slices")
    ap.add_argument("--step3_write_audio", action="store_true", help="Write individual WAV slices in Step-3")
    ap.add_argument("--step3_max_slices", type=int, default=1000, help="Limit number of slices")
    ap.add_argument("--step3_path", default=None, help="Working dir for Step-3 (optional)")

    # General
    ap.add_argument("--python", default=None, help="Path to Python interpreter (default: current)")

    args = ap.parse_args()

    wav_path = Path(args.wav).resolve()
    out_base = Path(args.out_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    py = args.python or _which_python()

    # --- Step 1: run sanity checker (creates per-run folder) ---
    step1_cmd = _with_unbuffered(py, Path(args.sanity_script)) + [
        str(wav_path),
        "--out_dir", str(out_base),
        "--mem_budget_mb", str(max(64, args.mem_budget_mb))
    ]
    if args.fs_hint is not None:
        step1_cmd += ["--fs_hint", str(args.fs_hint)]
    if args.fft_size is not None:
        step1_cmd += ["--fft_size", str(args.fft_size)]

    print("[RUN] Step-1:", " ".join(f'"{c}"' if " " in c else c for c in step1_cmd))
    rc1, out1 = _stream(step1_cmd, cwd=args.sanity_path)
    if rc1 != 0:
        sys.exit(f"[ERROR] Step-1 failed with code {rc1}")

    # Parse Step-1 JSON
    results1 = _extract_json(out1)
    if results1 is None:
        sys.exit("[ERROR] Could not parse Step-1 JSON from stdout")

    best_conv_step1 = results1.get("best_convention", "I+jQ")
    mapped_conv = _map_best_conv_to_step2(best_conv_step1)

    # Find the per-run folder Step-1 created
    run_dir = _extract_run_dir(out1, out_base, wav_path.stem)
    if run_dir is None or not run_dir.exists():
        # Fallback: newest dir matching run_<stem>_*
        candidates = sorted(
            (d for d in out_base.glob(f"run_{wav_path.stem}_*") if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True
        )
        run_dir = candidates[0] if candidates else None

    if run_dir is None:
        sys.exit("[ERROR] Could not locate Step-1 run directory. Check Step-1 output.")

    print(f"[OK] Using run directory: {run_dir}")
    print(f"[OK] Best I/Q convention from Step-1: {best_conv_step1}  ->  Step-2 uses '{mapped_conv}'")

    # --- Step 2: run analyzer into the SAME run_dir ---
    step2_cmd = _with_unbuffered(py, Path(args.step2_script)) + [
        str(wav_path),
        "--out", str(run_dir),
        "--conv", mapped_conv,
        "--nperseg", str(args.nperseg),
        "--overlap", str(args.overlap),
        "--prom_db", str(args.prom_db),
        "--cfar_k", str(args.cfar_k),
        "--cfar_guard", str(args.cfar_guard),
        "--cfar_train", str(args.cfar_train),
        "--max_duration", str(args.max_duration),
    ]
    if args.fs_hint is not None:
        step2_cmd += ["--fs_hint", str(args.fs_hint)]
    if args.force_full:
        step2_cmd += ["--force_full"]

    print("\n[RUN] Step-2:", " ".join(f'"{c}"' if " " in c else c for c in step2_cmd))
    rc2, out2 = _stream(step2_cmd, cwd=args.step2_path)
    if rc2 != 0:
        sys.exit(f"[ERROR] Step-2 failed with code {rc2}")

    # --- Step 3: run signal detection & slicing into SAME run_dir (always) ---
    step3_cmd = _with_unbuffered(py, Path(args.step3_script)) + [
        str(wav_path),
        "--out", str(run_dir),
        "--mem_budget_mb", str(max(64, args.step3_mem_budget_mb)),
        "--rms_win_ms", str(args.step3_rms_win_ms),
        "--thresh_dbfs", str(args.step3_thresh_dbfs),
        "--min_dur_ms", str(args.step3_min_dur_ms),
        "--gap_ms", str(args.step3_gap_ms),
        "--pad_ms", str(args.step3_pad_ms),
        "--max_slices", str(args.step3_max_slices),
    ]
    if args.fs_hint is not None:
        step3_cmd += ["--fs_hint", str(args.fs_hint)]
    if args.step3_write_audio:
        step3_cmd += ["--write_audio"]

    print("\n[RUN] Step-3:", " ".join(f'"{c}"' if " " in c else c for c in step3_cmd))
    rc3, out3 = _stream(step3_cmd, cwd=args.step3_path)
    if rc3 != 0:
        sys.exit(f"[ERROR] Step-3 failed with code {rc3}")

    print(f'\n[DONE] All outputs in: "{run_dir}"')

if __name__ == "__main__":
    main()

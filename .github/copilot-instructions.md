# Copilot Instructions for Sanity_Checker

## Project Overview
This repository implements a multi-stage Python pipeline for analyzing I/Q (In-phase/Quadrature) RF data from WAV files. The pipeline processes wideband radio frequency recordings to detect, slice, and classify radio signals, using both traditional signal processing and machine learning.

## Architecture & Data Flow
- **Pipeline Stages:**
  1. `1_sanity_check_iq.py`: Validates I/Q data, checks for clipping, determines I/Q convention, calculates IRR.
  2. `2_wideband_exploration.py`: Spectral analysis (PSD, waterfall), CFAR detection, IRR validation, visualization.
  3. `3_signal_detection_slicing.py`: Down-mixes carriers, filters, decimates, slices signals to HDF5.
  4. `4_feature_extraction.py`: Computes features (SNR, bandwidth, cumulants), engineered channels for ML.
  5. `6_insight_generation.py`: (Placeholder) Runs ML model for classification.
  6. `7_qa.py`: QA on predictions, conflict analysis, generates review reports.
- **Data Flow:**
  - Input: Stereo WAV files (I/Q samples)
  - Intermediate: HDF5 (signal clips), CSV/JSON (metadata, features)
  - Output: Classification results, QA/conflict reports

## Key Workflows
- **Batch Processing:** Use `run_iq_pipeline_v2.py` with `--batch_mode` for large-scale, memory-safe, checkpointed processing. Supports resume, memory limits, and HTML result indexing.
- **Single File:** Run `run_iq_pipeline_v2.py` for end-to-end processing of one file.
- **Stepwise Execution:** Each stage can be run independently for advanced workflows or debugging.
- **Parameter Tuning:**
  - `--prom_db` (Step 2): Raise to reduce false peaks in PSD.
  - `--cfar_k` (Step 2/3): Raise for stricter detection, less "speckle".
  - `--conv` (all): Use if Step 1 indicates I/Q swap needed.

## Conventions & Patterns
- All scripts require `--fs_hint` to override WAV header sample rate if needed.
- Outputs are written to `out_report/`, `slices_out/`, and step-specific folders.
- Use `--qc_plots` in Step 3 for per-carrier QC plots.
- Higher-order cumulant features are emphasized for modulation classification.
- Memory safety and checkpointing are prioritized for large batch jobs.

## Dependencies
- Core: `numpy`, `scipy.signal`, `soundfile`, `h5py`, `matplotlib`, `csv`, `json`
- Model: Pre-trained PyTorch checkpoint (see `Model/`)

## Examples
- Batch: `python run_iq_pipeline_v2.py /path/to/wavs --fs_hint 20000000 --batch_mode`
- Step 2: `python 2_wideband_exploration.py your.wav --fs_hint 20000000 --out out_report`
- Step 4: `python 4_feature_extraction.py --slices_h5 slices_out/slices.h5 --meta_json slices_out/meta.json --out_dir step4_out`

## References
- See `CLAUDE.md` for detailed command examples and parameter explanations.
- Key scripts: `run_iq_pipeline_v2.py`, `1_sanity_check_iq.py`, `2_wideband_exploration.py`, etc.
- Model artifacts: `Model/`
- Output: `out_report/`, `slices_out/`

---
For new features, follow the established stepwise pipeline pattern and ensure compatibility with batch processing and checkpointing. When in doubt, review `CLAUDE.md` for canonical usage patterns and parameter conventions.

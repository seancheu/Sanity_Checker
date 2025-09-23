# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based signal processing pipeline for analyzing I/Q (In-phase/Quadrature) RF data from WAV files. The system processes wideband radio frequency recordings to detect, slice, and classify radio signals using a multi-stage approach with machine learning.

## Commands

### Running the Complete Pipeline

Execute the 7-step analysis pipeline in order:

```bash
# Step 1: Sanity check I/Q data
python 1_sanity_check_iq.py your.wav --fs_hint 20000000

# Step 2: Wideband exploration - generates PSD, waterfall, occupancy plots
python 2_wideband_exploration.py your_20MHz.wav --fs_hint 20000000 --out out_report --nperseg 4096 --overlap 0.5 --prom_db 8 --cfar_k 3.0

# For large files (>60s), automatic optimization applies:
# - Smart sampling preserves spectral characteristics
# - Chunked CFAR processing for memory efficiency
# - Progress indicators show real-time status

# Performance options:
# --max_duration 30    # More aggressive sampling (30s target)
# --force_full         # Disable sampling (slower but full precision)

# Step 3A: Slice signals per carrier (whole file)
python 3_signal_detection_slicing.py your_20MHz.wav --fs_hint 20000000 --carriers_csv out_report/carriers.csv --mode carriers --win 16384 --hop_frac 0.5 --oversample 4.0 --min_bw 100000 --out slices_out

# Step 3B: Slice signals only when active (for bursty signals)
python 3_signal_detection_slicing.py your_20MHz.wav --fs_hint 20000000 --carriers_csv out_report/carriers.csv --mode bursts --nperseg 4096 --overlap 0.5 --cfar_k 3.0 --win 16384 --hop_frac 0.5 --oversample 4.0 --min_bw 100000 --out slices_out_bursty

# Step 4A: Extract features only (CSV)
python 4_feature_extraction.py --slices_h5 slices_out/slices.h5 --meta_json slices_out/meta.json --out_dir step4_out

# Step 4B: Extract features + engineered channels HDF5 for deep learning
python 4_feature_extraction.py --slices_h5 slices_out/slices.h5 --meta_json slices_out/meta.json --out_dir step4_out --emit_engineered_h5 --extras "amp,phase,dfreq,cosphi,sinphi,d_amp,d2phase,cum40,cum41,cum42"

# Step 6: Run inference with trained model (requires pre-trained checkpoint)
python 6_insight_generation.py --slices_h5 slices_out/slices.h5 --meta_json slices_out/meta.json --ckpt /path/to/your/best_model.pt --classes "OOK,BPSK,QPSK,8PSK,16QAM,64QAM,256QAM,AM-SSB-WC,AM-SSB-SC,FM,GMSK,OQPSK" --extra_ch "amp,phase,dfreq,cosphi,sinphi,d_amp,d2phase,cum40,cum41,cum42" --T_crop 16384 --blocks 10 --kernel 7 --rnn_hidden 384 --out_dir step6_out

# Step 7: Quality assurance and conflict analysis
python 7_qa.py --detections_csv step6_out/detections.csv --features_csv step4_out/features.csv --conf_thresh 0.70 --dominance_thresh 0.75 --out_dir step7_out
```

### Parameter Tuning

- If PSD marks too many tiny peaks: raise `--prom_db` to 10-12
- If waterfall mask is too "speckly": raise `--cfar_k` to 3.5-4.0 (stricter)
- If Step 1 indicates I/Q swapping needed: add `--conv I-Q` or `--conv Q+I` to subsequent steps

## Architecture

### Pipeline Stages

1. **Sanity Check** (`1_sanity_check_iq.py`): Validates I/Q data integrity, checks for clipping, determines optimal I/Q convention, and calculates image rejection ratio (IRR)

2. **Wideband Exploration** (`2_wideband_exploration.py`): Performs spectral analysis using Welch PSD and STFT waterfall, implements CFAR detection for occupancy analysis, and generates visualization outputs

3. **Signal Detection & Slicing** (`3_signal_detection_slicing.py`): Down-mixes detected carriers to baseband, applies low-pass filtering and decimation, creates windowed signal clips in HDF5 format

4. **Feature Extraction** (`4_feature_extraction.py`): Computes spectral features (SNR, bandwidth, spectral flatness), calculates higher-order cumulants for modulation classification, generates engineered channels for deep learning

5. **Insight Generation** (`6_insight_generation.py`): Applies trained TCN-BiLSTM model for signal classification (placeholder - actual implementation not present)

6. **Quality Assurance** (`7_qa.py`): Analyzes prediction confidence levels, identifies carrier-level conflicts, generates review reports for low-confidence detections

### Data Flow

- Input: Stereo WAV files with I/Q samples
- Intermediate: HDF5 files containing windowed signal clips, CSV files with carrier metadata and features
- Output: Classification results with confidence metrics, QA reports for manual review

### Key Algorithms

- **CFAR Detection**: 2D Constant False Alarm Rate using robust median/MAD statistics
- **Down-mixing**: Complex mixing with FIR low-pass filtering and polyphase decimation
- **Feature Engineering**: Spectral features, envelope statistics, and higher-order cumulants
- **Signal Processing**: Welch PSD estimation, STFT waterfalls, carrier frequency offset estimation

### Dependencies

Core libraries used throughout the pipeline:
- `numpy`, `scipy.signal` for signal processing
- `soundfile` for WAV I/O
- `h5py` for HDF5 data storage
- `matplotlib` for visualization
- `csv`, `json` for metadata handling

## Notes

- All scripts accept `--fs_hint` to override WAV header sample rate when incorrect
- Use `--qc_plots` flag in Step 3 to generate per-carrier quality control plots
- The pipeline supports both continuous carrier analysis and burst-mode detection
- Higher-order cumulant features are particularly useful for distinguishing modulation types
- CFAR parameters may need tuning based on signal characteristics and noise floor
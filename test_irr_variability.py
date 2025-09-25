#!/usr/bin/env python3
"""
IRR Variability Analysis Script

Tests how much IRR varies across different files, file segments, and conditions.

Usage:
    python test_irr_variability.py [file1.wav] [file2.wav] ... --fs_hint 20000000
    python test_irr_variability.py --test_single file.wav --fs_hint 20000000
"""

import argparse
import numpy as np
import soundfile as sf
import time
from pathlib import Path
import json
from datetime import datetime
import sys

# Import IRR calculation
sys.path.append('.')

# IRR calculation functions (from verify script)
EPSILON = 1e-12
DEFAULT_FFT_SIZE = 1 << 18

def calculate_adaptive_fft_size(num_samples: int, target_size: int = DEFAULT_FFT_SIZE) -> int:
    max_size = min(target_size, num_samples)
    if max_size < 2:
        return 1 << 12
    fft_size = 1 << (max_size.bit_length() - 1)
    return max(1 << 12, min(1 << 20, fft_size))

def irr_pos_neg_powers(complex_signal: np.ndarray, fft_size: int):
    if complex_signal.size < fft_size:
        pad = fft_size - complex_signal.size
        left = pad // 2
        right = pad - left
        complex_signal = np.pad(complex_signal, (left, right), mode='constant')
    else:
        start = (complex_signal.size - fft_size) // 2
        complex_signal = complex_signal[start:start+fft_size]

    X = np.fft.fftshift(np.fft.fft(complex_signal, n=fft_size))
    P = (np.abs(X) ** 2).astype(np.float64)
    mid = P.size // 2
    pos = float(P[mid+1:].sum() + EPSILON)
    neg = float(P[:mid].sum() + EPSILON)
    return pos, neg

def calculate_irr_db(xc: np.ndarray, fft_size: int = None) -> float:
    if fft_size is None:
        fft_size = calculate_adaptive_fft_size(len(xc))

    xc_dc = xc - np.mean(xc)
    pos, neg = irr_pos_neg_powers(xc_dc, fft_size)
    return 10.0 * np.log10(pos / (neg + EPSILON))

def create_complex_iq(I, Q, convention):
    I_dc = I - np.mean(I)
    Q_dc = Q - np.mean(Q)

    if convention == "I+jQ":
        return I_dc + 1j * Q_dc
    elif convention == "I-jQ":
        return I_dc - 1j * Q_dc
    elif convention == "Q+jI":
        return Q_dc + 1j * I_dc
    elif convention == "Q-jI":
        return Q_dc - 1j * I_dc
    else:
        raise ValueError(f"Unknown convention: {convention}")

def find_best_convention(I, Q):
    """Find best I/Q convention and return (convention, IRR)."""
    conventions = ["I+jQ", "I-jQ", "Q+jI", "Q-jI"]
    results = {}

    for conv in conventions:
        try:
            xc = create_complex_iq(I, Q, conv)
            irr_db = calculate_irr_db(xc)
            results[conv] = irr_db
        except Exception as e:
            results[conv] = float('-inf')

    best_conv = max(results.keys(), key=lambda k: results[k])
    return best_conv, results[best_conv], results

def analyze_single_file_segments(file_path, fs_hint, num_segments=10, segment_duration=5.0):
    """Analyze IRR variability within a single file by testing different segments."""
    print(f"\n=== Single File Segment Analysis: {Path(file_path).name} ===")

    # Load file info
    info = sf.info(file_path)
    fs = fs_hint if fs_hint else info.samplerate
    total_samples = info.frames
    total_duration = total_samples / fs

    segment_samples = int(segment_duration * fs)

    if total_samples < segment_samples:
        print(f"File too short ({total_duration:.1f}s) for segment analysis")
        return None

    print(f"File duration: {total_duration:.1f}s")
    print(f"Analyzing {num_segments} segments of {segment_duration}s each")

    results = {
        'file': str(file_path),
        'total_duration': total_duration,
        'segment_duration': segment_duration,
        'segments': []
    }

    # Test segments throughout the file
    step = max(1, (total_samples - segment_samples) // (num_segments - 1))

    for i in range(num_segments):
        start_sample = min(i * step, total_samples - segment_samples)
        end_sample = start_sample + segment_samples

        # Read segment
        with sf.SoundFile(file_path) as f:
            f.seek(start_sample)
            x = f.read(segment_samples, always_2d=True)

        if x.shape[1] < 2:
            continue

        I, Q = x[:, 0].astype(np.float32), x[:, 1].astype(np.float32)

        # Analyze segment
        best_conv, best_irr, all_irr = find_best_convention(I, Q)

        segment_result = {
            'segment': i + 1,
            'start_time': start_sample / fs,
            'end_time': end_sample / fs,
            'best_convention': best_conv,
            'best_irr_db': best_irr,
            'all_conventions': all_irr
        }

        results['segments'].append(segment_result)

        print(f"Segment {i+1:2d} ({start_sample/fs:6.1f}-{end_sample/fs:6.1f}s): "
              f"{best_conv} = {best_irr:+7.2f} dB")

    # Calculate statistics
    irr_values = [s['best_irr_db'] for s in results['segments']]
    conventions = [s['best_convention'] for s in results['segments']]

    if irr_values:
        results['statistics'] = {
            'mean_irr': np.mean(irr_values),
            'std_irr': np.std(irr_values),
            'min_irr': np.min(irr_values),
            'max_irr': np.max(irr_values),
            'range_irr': np.max(irr_values) - np.min(irr_values),
            'most_common_convention': max(set(conventions), key=conventions.count),
            'convention_consistency': conventions.count(conventions[0]) / len(conventions)
        }

        stats = results['statistics']
        print(f"\nSegment Statistics:")
        print(f"  IRR Range: {stats['min_irr']:+.1f} to {stats['max_irr']:+.1f} dB (span: {stats['range_irr']:.1f} dB)")
        print(f"  Mean ± Std: {stats['mean_irr']:+.1f} ± {stats['std_irr']:.1f} dB")
        print(f"  Most common convention: {stats['most_common_convention']}")
        print(f"  Convention consistency: {stats['convention_consistency']:.1%}")

        # Classify variability
        if stats['range_irr'] < 2.0:
            variability = "LOW"
        elif stats['range_irr'] < 10.0:
            variability = "MODERATE"
        else:
            variability = "HIGH"

        print(f"  IRR Variability: {variability}")
        results['variability_assessment'] = variability

    return results

def analyze_multiple_files(file_paths, fs_hint):
    """Analyze IRR across multiple different files."""
    print(f"\n=== Multiple File Analysis ===")

    results = {
        'analysis_time': datetime.now().isoformat(),
        'files': []
    }

    for file_path in file_paths:
        print(f"\nAnalyzing: {Path(file_path).name}")

        try:
            # Load first 10M samples for consistency
            x, fs_hdr = sf.read(file_path, frames=10_000_000, always_2d=True)
            fs = fs_hint if fs_hint else fs_hdr

            if x.shape[1] < 2:
                print(f"  Skipping: not stereo I/Q")
                continue

            I, Q = x[:, 0].astype(np.float32), x[:, 1].astype(np.float32)
            duration = len(I) / fs

            # Find best convention
            best_conv, best_irr, all_irr = find_best_convention(I, Q)

            file_result = {
                'file': str(file_path),
                'duration_analyzed': duration,
                'best_convention': best_conv,
                'best_irr_db': best_irr,
                'all_conventions': all_irr
            }

            results['files'].append(file_result)

            print(f"  Duration: {duration:.1f}s, Best: {best_conv} = {best_irr:+.1f} dB")

            # Show all conventions for comparison
            for conv, irr in all_irr.items():
                marker = " <--" if conv == best_conv else ""
                print(f"    {conv}: {irr:+7.2f} dB{marker}")

        except Exception as e:
            print(f"  Error analyzing {file_path}: {e}")
            continue

    # Cross-file statistics
    if len(results['files']) > 1:
        irr_values = [f['best_irr_db'] for f in results['files']]
        conventions = [f['best_convention'] for f in results['files']]

        results['cross_file_stats'] = {
            'num_files': len(results['files']),
            'mean_irr': np.mean(irr_values),
            'std_irr': np.std(irr_values),
            'min_irr': np.min(irr_values),
            'max_irr': np.max(irr_values),
            'range_irr': np.max(irr_values) - np.min(irr_values),
            'convention_distribution': {conv: conventions.count(conv) for conv in set(conventions)}
        }

        stats = results['cross_file_stats']
        print(f"\nCross-File Statistics:")
        print(f"  Files analyzed: {stats['num_files']}")
        print(f"  IRR Range: {stats['min_irr']:+.1f} to {stats['max_irr']:+.1f} dB (span: {stats['range_irr']:.1f} dB)")
        print(f"  Mean ± Std: {stats['mean_irr']:+.1f} ± {stats['std_irr']:.1f} dB")
        print(f"  Convention distribution: {stats['convention_distribution']}")

        # File-to-file variability assessment
        if stats['range_irr'] < 5.0:
            variability = "LOW"
        elif stats['range_irr'] < 20.0:
            variability = "MODERATE"
        else:
            variability = "HIGH"

        print(f"  Cross-file variability: {variability}")
        results['cross_file_variability'] = variability

    return results

def test_sampling_effects(file_path, fs_hint):
    """Test how different sampling strategies affect IRR."""
    print(f"\n=== Sampling Effects Analysis: {Path(file_path).name} ===")

    # Load full data
    x, fs_hdr = sf.read(file_path, frames=20_000_000, always_2d=True)  # Up to 20M samples
    fs = fs_hint if fs_hint else fs_hdr

    I, Q = x[:, 0].astype(np.float32), x[:, 1].astype(np.float32)

    # Find best convention on full data
    best_conv, baseline_irr, _ = find_best_convention(I, Q)
    print(f"Baseline (full data): {best_conv} = {baseline_irr:+.2f} dB")

    sampling_results = []

    # Test different sampling rates
    for factor in [1, 2, 3, 4, 5, 8, 10, 16]:
        sampled_I = I[::factor]
        sampled_Q = Q[::factor]

        xc = create_complex_iq(sampled_I, sampled_Q, best_conv)
        sampled_irr = calculate_irr_db(xc)
        difference = sampled_irr - baseline_irr

        result = {
            'sampling_factor': factor,
            'effective_fs_mhz': fs / factor / 1e6,
            'irr_db': sampled_irr,
            'difference_db': difference
        }
        sampling_results.append(result)

        print(f"1:{factor:2d} sampling ({fs/factor/1e6:5.1f} MHz): {sampled_irr:+7.2f} dB (D{difference:+6.2f} dB)")

    return {
        'baseline_irr': baseline_irr,
        'baseline_convention': best_conv,
        'sampling_results': sampling_results
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze IRR variability across files and conditions")
    parser.add_argument("files", nargs="*", help="WAV files to analyze")
    parser.add_argument("--fs_hint", type=float, help="Sample rate override")
    parser.add_argument("--test_single", help="Test single file segment variability")
    parser.add_argument("--test_sampling", help="Test sampling effects on single file")
    parser.add_argument("--segments", type=int, default=10, help="Number of segments for single file test")
    parser.add_argument("--segment_duration", type=float, default=5.0, help="Segment duration in seconds")

    args = parser.parse_args()

    print("IRR Variability Analysis")
    print("=" * 50)

    all_results = {}

    # Single file segment analysis
    if args.test_single:
        result = analyze_single_file_segments(
            args.test_single, args.fs_hint,
            args.segments, args.segment_duration
        )
        if result:
            all_results['single_file_segments'] = result

    # Sampling effects analysis
    if args.test_sampling:
        result = test_sampling_effects(args.test_sampling, args.fs_hint)
        all_results['sampling_effects'] = result

    # Multiple file analysis
    if args.files:
        result = analyze_multiple_files(args.files, args.fs_hint)
        all_results['multiple_files'] = result

    # Save comprehensive results
    if all_results:
        output_file = Path("irr_variability_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Summary insights
        print(f"\n=== SUMMARY INSIGHTS ===")

        # Single file variability
        if 'single_file_segments' in all_results:
            seg_var = all_results['single_file_segments'].get('variability_assessment', 'UNKNOWN')
            print(f"Within-file variability: {seg_var}")

        # Cross-file variability
        if 'multiple_files' in all_results:
            cross_var = all_results['multiple_files'].get('cross_file_variability', 'UNKNOWN')
            print(f"Cross-file variability: {cross_var}")

        # Sampling impact
        if 'sampling_effects' in all_results:
            sampling = all_results['sampling_effects']['sampling_results']
            worst_degradation = min(s['difference_db'] for s in sampling)
            print(f"Worst sampling degradation: {worst_degradation:+.1f} dB")

    else:
        print("No analysis performed. Use --test_single, --test_sampling, or provide file arguments.")

if __name__ == "__main__":
    main()
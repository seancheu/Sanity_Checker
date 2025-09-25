#!/usr/bin/env python3
"""
I/Q Convention Verification Script

This script helps verify the correct I/Q convention mapping between Step 1 and Step 2
by testing all possible combinations and comparing IRR results.

Usage:
    python verify_iq_convention.py your.wav --fs_hint 20000000
"""

import argparse
import numpy as np
import soundfile as sf
import time
from pathlib import Path

# Import IRR calculation from existing scripts
import sys
sys.path.append('.')

# IRR calculation functions (copied from existing scripts)
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

def create_complex_iq(I, Q, convention):
    """Create complex I/Q signal using specified convention."""
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

def step1_style_analysis(I, Q):
    """Replicate Step 1's I/Q convention analysis."""
    conventions = ["I+jQ", "I-jQ", "Q+jI", "Q-jI"]
    results = {}

    print("=== Step 1 Style Analysis ===")
    for conv in conventions:
        try:
            xc = create_complex_iq(I, Q, conv)
            irr_db = calculate_irr_db(xc)
            results[conv] = irr_db
            print(f"{conv:>6}: {irr_db:+7.2f} dB")
        except Exception as e:
            print(f"{conv:>6}: ERROR - {e}")
            results[conv] = float('-inf')

    best_conv = max(results.keys(), key=lambda k: results[k])
    best_irr = results[best_conv]

    print(f"Best: {best_conv} ({best_irr:+.2f} dB)")
    return best_conv, best_irr, results

def step2_style_analysis(I, Q):
    """Test Step 2's I/Q convention interpretations."""
    # Step 2 conventions (different format)
    conventions = {
        "I+Q": lambda i, q: i + 1j*q,
        "I-Q": lambda i, q: i - 1j*q,
        "Q+I": lambda i, q: q + 1j*i,
        "Q-I": lambda i, q: q - 1j*i
    }

    results = {}

    print("\n=== Step 2 Style Analysis ===")
    for conv_name, conv_func in conventions.items():
        try:
            # Remove DC (Step 2 style)
            I_dc = I - np.mean(I)
            Q_dc = Q - np.mean(Q)
            xc = conv_func(I_dc, Q_dc)
            irr_db = calculate_irr_db(xc)
            results[conv_name] = irr_db
            print(f"{conv_name:>6}: {irr_db:+7.2f} dB")
        except Exception as e:
            print(f"{conv_name:>6}: ERROR - {e}")
            results[conv_name] = float('-inf')

    best_conv = max(results.keys(), key=lambda k: results[k])
    best_irr = results[best_conv]

    print(f"Best: {best_conv} ({best_irr:+.2f} dB)")
    return best_conv, best_irr, results

def verify_mapping(step1_best, step2_best, step1_irr, step2_irr):
    """Verify if the current mapping is correct."""
    current_mapping = {
        "I+jQ": "I+Q",
        "I-jQ": "I-Q",
        "Q+jI": "Q+I",
        "Q-jI": "Q-I"
    }

    print(f"\n=== Convention Mapping Verification ===")
    print(f"Step 1 Best: {step1_best} ({step1_irr:+.2f} dB)")
    print(f"Step 2 Best: {step2_best} ({step2_irr:+.2f} dB)")
    print(f"IRR Difference: {step2_irr - step1_irr:+.2f} dB")

    expected_step2 = current_mapping.get(step1_best, "UNKNOWN")
    print(f"Current mapping: {step1_best} -> {expected_step2}")

    if expected_step2 == step2_best:
        print("[OK] Mapping appears correct!")
        if abs(step2_irr - step1_irr) < 3.0:
            print("[OK] IRR values are consistent")
            return True, "Mapping verified - no changes needed"
        else:
            print("[WARN] IRR values differ significantly")
            return False, f"IRR mismatch: {abs(step2_irr - step1_irr):.1f} dB difference"
    else:
        print("[ERROR] Mapping mismatch detected!")
        print(f"Expected: {step1_best} -> {expected_step2}")
        print(f"Actual best: {step1_best} -> {step2_best}")
        return False, f"Convention mapping should be: {step1_best} -> {step2_best}"

def test_sampling_impact(I, Q, best_step1_conv, reduction_factors=[1, 2, 4, 8]):
    """Test how sampling affects IRR measurements."""
    print(f"\n=== Sampling Impact Analysis ===")
    print(f"Testing convention: {best_step1_conv}")

    baseline_xc = create_complex_iq(I, Q, best_step1_conv)
    baseline_irr = calculate_irr_db(baseline_xc)

    print(f"Full data:  {baseline_irr:+7.2f} dB (baseline)")

    for factor in reduction_factors[1:]:
        # Systematic sampling (like Step 2 does)
        sampled_I = I[::factor]
        sampled_Q = Q[::factor]

        sampled_xc = create_complex_iq(sampled_I, sampled_Q, best_step1_conv)
        sampled_irr = calculate_irr_db(sampled_xc)
        difference = sampled_irr - baseline_irr

        print(f"1:{factor:2d} sample: {sampled_irr:+7.2f} dB (D{difference:+5.2f} dB)")

def main():
    parser = argparse.ArgumentParser(description="Verify I/Q convention mapping between pipeline steps")
    parser.add_argument("wav", help="Path to I/Q WAV file")
    parser.add_argument("--fs_hint", type=float, help="Sample rate override")
    parser.add_argument("--max_samples", type=int, default=10_000_000, help="Max samples to analyze")
    args = parser.parse_args()

    print(f"I/Q Convention Verification")
    print(f"File: {args.wav}")
    print("=" * 50)

    # Load data
    print("Loading data...")
    x, fs_hdr = sf.read(args.wav, always_2d=True)
    fs = args.fs_hint if args.fs_hint else fs_hdr

    if x.shape[1] < 2:
        raise ValueError("Need stereo I/Q data")

    # Limit data size for faster processing
    if len(x) > args.max_samples:
        print(f"Using first {args.max_samples/1e6:.1f}M samples for analysis")
        x = x[:args.max_samples]

    I = x[:, 0].astype(np.float32)
    Q = x[:, 1].astype(np.float32)

    print(f"Samples: {len(I):,} @ {fs/1e6:.1f} MHz")
    print(f"Duration: {len(I)/fs:.2f} seconds")

    # Analyze both ways
    step1_best, step1_irr, step1_results = step1_style_analysis(I, Q)
    step2_best, step2_irr, step2_results = step2_style_analysis(I, Q)

    # Verify mapping
    is_correct, message = verify_mapping(step1_best, step2_best, step1_irr, step2_irr)

    # Test sampling impact
    test_sampling_impact(I, Q, step1_best)

    # Summary and recommendations
    print(f"\n=== SUMMARY & RECOMMENDATIONS ===")
    print(f"Status: {'PASS' if is_correct else 'FAIL'}")
    print(f"Message: {message}")

    if not is_correct:
        print(f"\nRecommended fix:")
        print(f"Update pipeline mapping: '{step1_best}' -> '{step2_best}'")
        print(f"Expected IRR improvement: ~{abs(step2_irr - step1_irr):.1f} dB")

    # Output for pipeline integration
    result = {
        "verification_passed": is_correct,
        "step1_best": step1_best,
        "step1_irr": step1_irr,
        "step2_best": step2_best,
        "step2_irr": step2_irr,
        "recommended_mapping": {step1_best: step2_best} if not is_correct else None,
        "message": message
    }

    # Save results
    output_file = Path("iq_convention_verification.json")
    import json
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
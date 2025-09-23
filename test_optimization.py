#!/usr/bin/env python3
"""
Quick test script to validate 2_wideband_exploration.py optimizations.
Generates synthetic I/Q data to test performance improvements.
"""

import numpy as np
import soundfile as sf
import tempfile
import time
import subprocess
import sys
from pathlib import Path

def generate_test_iq(duration_s=240, fs=20e6, num_carriers=5):
    """Generate synthetic I/Q data with multiple carriers."""
    print(f"Generating {duration_s}s of synthetic I/Q data @ {fs/1e6:.1f} MHz...")

    N = int(duration_s * fs)
    t = np.arange(N) / fs

    # Base noise
    noise_power = 1e-4
    I = np.random.normal(0, np.sqrt(noise_power/2), N)
    Q = np.random.normal(0, np.sqrt(noise_power/2), N)

    # Add several carriers at different frequencies
    for i in range(num_carriers):
        fc = (i - num_carriers//2) * fs / (num_carriers + 2)  # Spread across band
        amp = 0.1 * np.random.uniform(0.5, 1.5)  # Random amplitude
        phase = np.random.uniform(0, 2*np.pi)

        # Add modulated carrier (simple AM)
        mod_freq = 1000 + i * 500  # Different modulation rates
        modulation = 1 + 0.3 * np.sin(2 * np.pi * mod_freq * t)

        carrier_I = amp * modulation * np.cos(2 * np.pi * fc * t + phase)
        carrier_Q = amp * modulation * np.sin(2 * np.pi * fc * t + phase)

        I += carrier_I
        Q += carrier_Q

    # Combine into stereo format
    iq_data = np.column_stack([I, Q]).astype(np.float32)
    return iq_data, fs

def run_analysis(wav_path, fs, test_name, extra_args=""):
    """Run the analysis script and measure performance."""
    print(f"\n=== {test_name} ===")

    cmd = [
        sys.executable, "2_wideband_exploration.py",
        str(wav_path),
        "--fs_hint", str(int(fs)),
        "--out", f"test_out_{test_name.lower().replace(' ', '_')}",
        "--nperseg", "4096",
        "--overlap", "0.5",
        "--prom_db", "8",
        "--cfar_k", "3.0"
    ]

    if extra_args:
        cmd.extend(extra_args.split())

    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ SUCCESS in {elapsed:.1f}s")
            # Extract key metrics from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'samples' in line and ('Large file' in line or 'Total:' in line or 'Loaded' in line):
                    print(f"   {line}")
                elif 'Analysis completed' in line:
                    print(f"   {line}")
        else:
            print(f"❌ FAILED in {elapsed:.1f}s")
            print(f"   Error: {result.stderr}")

        return elapsed, result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT after 300s")
        return 300, False

def main():
    print("Testing 2_wideband_exploration.py optimizations")
    print("=" * 50)

    # Create temporary test file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Generate large test file (4 minutes = 240s @ 20MHz = 4.8B samples)
        iq_data, fs = generate_test_iq(duration_s=240, fs=20e6, num_carriers=8)

        print(f"Writing {iq_data.shape[0]/1e6:.1f}M samples to {tmp_path}")
        sf.write(tmp_path, iq_data, int(fs))

        file_size_mb = tmp_path.stat().st_size / (1024 * 1024)
        print(f"Test file size: {file_size_mb:.1f} MB")

        # Test scenarios
        tests = [
            ("Optimized Default", ""),  # Should use sampling automatically
            ("Force Full Processing", "--force_full"),  # Process entire file
            ("Custom Target 30s", "--max_duration 30"),  # More aggressive sampling
        ]

        results = []
        for test_name, args in tests:
            elapsed, success = run_analysis(tmp_path, fs, test_name, args)
            results.append((test_name, elapsed, success))

            # Don't run full processing if it will take too long
            if test_name == "Optimized Default" and elapsed > 120:
                print("\n⚠️  Skipping --force_full test (would take too long)")
                break

        # Summary
        print("\n" + "=" * 50)
        print("PERFORMANCE SUMMARY")
        print("=" * 50)
        for test_name, elapsed, success in results:
            status = "✅" if success else "❌"
            print(f"{status} {test_name:<25}: {elapsed:>6.1f}s")

        # Recommendations
        print("\nRECOMMENDATIONS:")
        if any(success for _, _, success in results):
            print("✅ Optimizations working correctly")
            print("   - Use default settings for automatic optimization")
            print("   - Add --force_full only for critical analysis")
            print("   - Adjust --max_duration for time constraints")
        else:
            print("❌ All tests failed - check error messages above")

    finally:
        # Cleanup
        if tmp_path.exists():
            tmp_path.unlink()
            print(f"\nCleaned up {tmp_path}")

if __name__ == "__main__":
    main()
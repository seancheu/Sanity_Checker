#!/usr/bin/env python3
"""
Batch processor for multiple large RF64 WAV files (22GB+ each)

Designed to handle 50+ files of 22GB each with:
- Sequential processing (one file at a time)
- Memory-safe chunked processing within each file
- Automatic memory cleanup between files
- Progress tracking and resumability
- Safe error handling with per-file isolation

Usage:
    python batch_processor.py /path/to/folder/with/wav/files --fs_hint 20000000 --chunk_duration 30
    python batch_processor.py file1.wav file2.wav file3.wav --fs_hint 20000000 --chunk_duration 60
"""

import argparse
import gc
import json
import logging
import os
import psutil
import shutil
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Handles sequential processing of multiple large WAV files"""

    def __init__(self, output_dir: Path, fs_hint: float, chunk_duration: float = 30.0,
                 memory_limit_gb: float = 8.0, python_exe: str = None):
        self.output_dir = Path(output_dir)
        self.fs_hint = fs_hint
        self.chunk_duration = chunk_duration
        self.memory_limit_gb = memory_limit_gb
        self.python_exe = python_exe or sys.executable

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.progress_file = self.output_dir / "batch_progress.json"
        self.completed_files = set()
        self.failed_files = set()

        # Load existing progress
        self._load_progress()

    def _load_progress(self):
        """Load progress from previous runs"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.completed_files = set(progress.get('completed', []))
                    self.failed_files = set(progress.get('failed', []))
                logger.info(f"Resuming: {len(self.completed_files)} completed, {len(self.failed_files)} failed")
        except Exception as e:
            logger.warning(f"Could not load progress: {e}")

    def _save_progress(self):
        """Save current progress"""
        try:
            progress = {
                'completed': list(self.completed_files),
                'failed': list(self.failed_files),
                'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")

    def _check_memory(self) -> Tuple[float, float]:
        """Check current memory usage"""
        try:
            memory = psutil.virtual_memory()
            used_gb = (memory.total - memory.available) / (1024**3)
            available_gb = memory.available / (1024**3)
            return used_gb, available_gb
        except:
            return 0.0, 8.0  # fallback values

    def _cleanup_memory(self):
        """Force memory cleanup"""
        gc.collect()
        try:
            # Additional cleanup for NumPy/SciPy arrays
            import numpy as np
            if hasattr(np, '_cleanup'):
                np._cleanup()
        except:
            pass

    def _wait_for_memory(self, required_gb: float = 4.0, timeout: int = 300):
        """Wait for sufficient memory to be available"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            used_gb, available_gb = self._check_memory()

            if available_gb >= required_gb:
                logger.info(f"Memory available: {available_gb:.1f}GB (required: {required_gb:.1f}GB)")
                return True

            logger.warning(f"Insufficient memory: {available_gb:.1f}GB available, need {required_gb:.1f}GB")
            logger.info("Waiting for memory to clear...")

            self._cleanup_memory()
            time.sleep(10)

        logger.error(f"Timeout waiting for memory after {timeout}s")
        return False

    def _get_file_chunks(self, file_path: Path) -> List[Tuple[float, float]]:
        """Calculate time chunks for a file"""
        try:
            # Get file duration from header inspection
            import soundfile as sf
            info = sf.info(file_path)
            duration_s = info.frames / self.fs_hint

            chunks = []
            start_time = 0.0

            while start_time < duration_s:
                end_time = min(start_time + self.chunk_duration, duration_s)
                chunks.append((start_time, end_time))
                start_time = end_time

            logger.info(f"File {file_path.name}: {duration_s:.1f}s → {len(chunks)} chunks of {self.chunk_duration:.1f}s")
            return chunks

        except Exception as e:
            logger.error(f"Could not analyze file {file_path}: {e}")
            # Fallback: assume full file needs processing
            return [(0.0, self.chunk_duration)]

    def _process_file_chunk(self, file_path: Path, start_time: float, end_time: float,
                          chunk_idx: int, total_chunks: int) -> bool:
        """Process a single chunk of a file"""
        try:
            # Wait for memory availability
            if not self._wait_for_memory(required_gb=4.0):
                raise RuntimeError("Insufficient memory for chunk processing")

            chunk_output_dir = self.output_dir / file_path.stem / f"chunk_{chunk_idx:04d}"
            chunk_output_dir.mkdir(parents=True, exist_ok=True)

            # Build command for chunk processing
            cmd = [
                self.python_exe,
                "run_iq_pipeline_v2.py",
                str(file_path),
                "--fs_hint", str(self.fs_hint),
                "--out_dir", str(chunk_output_dir),
                "--max_duration", str(end_time - start_time + 5),  # +5s buffer
                "--adaptive_sampling",
                "--skip_step3b",  # Skip most memory-intensive step for batch processing
                "--timeout", "3600"  # 1 hour timeout per chunk
            ]

            logger.info(f"Processing chunk {chunk_idx+1}/{total_chunks}: {start_time:.1f}-{end_time:.1f}s")

            # Execute with timeout and memory monitoring
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Monitor process
            output_lines = []
            while True:
                line = process.stdout.readline()
                if line:
                    output_lines.append(line)
                    if len(output_lines) % 10 == 0:  # Every 10 lines, check memory
                        used_gb, available_gb = self._check_memory()
                        if available_gb < 1.0:  # Less than 1GB available
                            logger.warning("Low memory detected, terminating chunk")
                            process.terminate()
                            return False

                if process.poll() is not None:
                    break

            return_code = process.wait()

            if return_code == 0:
                logger.info(f"Chunk {chunk_idx+1}/{total_chunks} completed successfully")

                # Save chunk info
                chunk_info = {
                    'file': str(file_path),
                    'chunk_idx': chunk_idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'return_code': return_code
                }

                with open(chunk_output_dir / "chunk_info.json", 'w') as f:
                    json.dump(chunk_info, f, indent=2)

                return True
            else:
                logger.error(f"Chunk {chunk_idx+1}/{total_chunks} failed with return code {return_code}")
                return False

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx+1}/{total_chunks}: {e}")
            return False
        finally:
            # Always cleanup after chunk
            self._cleanup_memory()

    def process_file(self, file_path: Path) -> bool:
        """Process a single large file using chunked approach"""
        logger.info(f"Starting file: {file_path.name} ({file_path.stat().st_size / (1024**3):.1f}GB)")

        try:
            # Check if already completed
            if str(file_path) in self.completed_files:
                logger.info(f"File {file_path.name} already completed, skipping")
                return True

            # Calculate chunks
            chunks = self._get_file_chunks(file_path)

            successful_chunks = 0
            failed_chunks = 0

            for chunk_idx, (start_time, end_time) in enumerate(chunks):
                success = self._process_file_chunk(
                    file_path, start_time, end_time, chunk_idx, len(chunks)
                )

                if success:
                    successful_chunks += 1
                else:
                    failed_chunks += 1

                # Save progress after each chunk
                self._save_progress()

                # Memory check between chunks
                used_gb, available_gb = self._check_memory()
                logger.info(f"Memory status: {used_gb:.1f}GB used, {available_gb:.1f}GB available")

                if available_gb < 2.0:
                    logger.warning("Low memory between chunks, forcing cleanup")
                    self._cleanup_memory()
                    time.sleep(5)

            # File completion status
            success_rate = successful_chunks / len(chunks)

            if success_rate >= 0.8:  # 80% success rate threshold
                logger.info(f"File {file_path.name} completed: {successful_chunks}/{len(chunks)} chunks successful")
                self.completed_files.add(str(file_path))
                return True
            else:
                logger.error(f"File {file_path.name} failed: only {successful_chunks}/{len(chunks)} chunks successful")
                self.failed_files.add(str(file_path))
                return False

        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")
            self.failed_files.add(str(file_path))
            return False

    def process_batch(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Process multiple files sequentially"""
        logger.info(f"Starting batch processing: {len(file_paths)} files")

        start_time = time.time()
        successful_files = 0
        failed_files = 0

        for idx, file_path in enumerate(file_paths, 1):
            logger.info(f"=== File {idx}/{len(file_paths)}: {file_path.name} ===")

            # Pre-processing memory check
            used_gb, available_gb = self._check_memory()
            logger.info(f"Pre-processing memory: {used_gb:.1f}GB used, {available_gb:.1f}GB available")

            if available_gb < 3.0:
                logger.warning("Low memory before file processing, waiting...")
                if not self._wait_for_memory(required_gb=3.0):
                    logger.error(f"Skipping {file_path.name} due to memory constraints")
                    failed_files += 1
                    continue

            success = self.process_file(file_path)

            if success:
                successful_files += 1
            else:
                failed_files += 1

            # Save progress after each file
            self._save_progress()

            # Inter-file cleanup and status
            self._cleanup_memory()
            elapsed_time = time.time() - start_time
            files_remaining = len(file_paths) - idx

            if idx > 0:
                avg_time_per_file = elapsed_time / idx
                eta_seconds = avg_time_per_file * files_remaining
                eta_hours = eta_seconds / 3600

                logger.info(f"Progress: {idx}/{len(file_paths)} files ({successful_files} success, {failed_files} failed)")
                logger.info(f"ETA: {eta_hours:.1f} hours ({eta_seconds/60:.1f} minutes)")

        # Final summary
        total_time = time.time() - start_time

        summary = {
            'total_files': len(file_paths),
            'successful_files': successful_files,
            'failed_files': failed_files,
            'success_rate': successful_files / len(file_paths) if file_paths else 0,
            'total_time_hours': total_time / 3600,
            'avg_time_per_file_minutes': (total_time / len(file_paths) / 60) if file_paths else 0
        }

        logger.info(f"Batch processing completed:")
        logger.info(f"  Total files: {summary['total_files']}")
        logger.info(f"  Successful: {summary['successful_files']}")
        logger.info(f"  Failed: {summary['failed_files']}")
        logger.info(f"  Success rate: {summary['success_rate']*100:.1f}%")
        logger.info(f"  Total time: {summary['total_time_hours']:.1f} hours")
        logger.info(f"  Avg per file: {summary['avg_time_per_file_minutes']:.1f} minutes")

        # Save final summary
        summary_file = self.output_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

def find_wav_files(input_paths: List[str]) -> List[Path]:
    """Find all WAV files from input paths (files or directories)"""
    wav_files = []

    for input_path in input_paths:
        path = Path(input_path)

        if path.is_file():
            if path.suffix.lower() in ['.wav', '.wave']:
                wav_files.append(path)
            else:
                logger.warning(f"Skipping non-WAV file: {path}")
        elif path.is_dir():
            # Find all WAV files in directory
            found_files = []
            for ext in ['*.wav', '*.WAV', '*.wave', '*.WAVE']:
                found_files.extend(path.glob(ext))

            logger.info(f"Found {len(found_files)} WAV files in {path}")
            wav_files.extend(found_files)
        else:
            logger.error(f"Invalid path: {path}")

    # Sort by name for consistent processing order
    wav_files.sort(key=lambda x: x.name.lower())

    logger.info(f"Total WAV files to process: {len(wav_files)}")
    for i, f in enumerate(wav_files[:10], 1):  # Show first 10
        size_gb = f.stat().st_size / (1024**3)
        logger.info(f"  {i:2d}. {f.name} ({size_gb:.1f}GB)")

    if len(wav_files) > 10:
        logger.info(f"  ... and {len(wav_files) - 10} more files")

    return wav_files

def main():
    parser = argparse.ArgumentParser(
        description="Batch processor for multiple large RF64 WAV files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("input_paths", nargs='+',
                       help="WAV files or directories containing WAV files")
    parser.add_argument("--fs_hint", type=float, required=True,
                       help="Sample rate hint (Hz) for all files")
    parser.add_argument("--chunk_duration", type=float, default=30.0,
                       help="Duration (seconds) for each processing chunk")
    parser.add_argument("--output_dir", default="batch_output",
                       help="Output directory for all results")
    parser.add_argument("--memory_limit_gb", type=float, default=8.0,
                       help="Memory limit (GB) before waiting/cleanup")
    parser.add_argument("--python", default=None,
                       help="Python interpreter path")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous batch processing")

    args = parser.parse_args()

    try:
        # Find all WAV files
        wav_files = find_wav_files(args.input_paths)

        if not wav_files:
            logger.error("No WAV files found!")
            return 1

        # Initialize batch processor
        processor = BatchProcessor(
            output_dir=args.output_dir,
            fs_hint=args.fs_hint,
            chunk_duration=args.chunk_duration,
            memory_limit_gb=args.memory_limit_gb,
            python_exe=args.python
        )

        # Process all files
        summary = processor.process_batch(wav_files)

        if summary['success_rate'] >= 0.8:
            logger.info("Batch processing completed successfully!")
            return 0
        else:
            logger.error("Batch processing completed with significant failures!")
            return 1

    except KeyboardInterrupt:
        logger.info("Batch processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
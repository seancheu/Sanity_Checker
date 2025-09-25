#!/usr/bin/env python3
"""
Hardened version of the I/Q pipeline runner with comprehensive security fixes.

Main runner that orchestrates (all steps enabled by default):
  1) 1_sanity_check_iq.py  (streaming, full-file)
  2) 2_wideband_exploration.py     (PSD, waterfall, CFAR, carriers)
  3) 3_signal_detection_slicing.py (streaming RMS-based detection & optional slicing)
  3B) Original signal slicing script (creates HDF5 slices for ML)
  4) 4_feature_extraction.py (extract features from slices)
  5) 5_inference.py (wideband classification inference)

By default, runs all steps 1-5. Use --only_stepX or --skip_stepX to control execution.

Security improvements:
- Input validation and sanitization to prevent command injection
- Path traversal protection with boundary validation
- Subprocess security controls with timeouts and resource limits
- Comprehensive error handling with retry logic
- Resource monitoring and cleanup on failure
- Secure logging without information disclosure

Usage:
  # Full pipeline (default - all steps):
  python run_iq_pipeline_v2.py "C:\path\to\your.wav" --fs_hint 20000000

  # Run only specific steps:
  python run_iq_pipeline_v2.py "C:\path\to\your.wav" --fs_hint 20000000 --only_step2

  # Skip certain steps:
  python run_iq_pipeline_v2.py "C:\path\to\your.wav" --fs_hint 20000000 --skip_step5

  # Run inference only (requires existing model and labels):
  python run_iq_pipeline_v2.py "C:\path\to\your.wav" --fs_hint 20000000 --only_step1 --step5_auto_vad
"""

import argparse
import atexit
import gc
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import glob

# Configure secure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global cleanup registry
cleanup_dirs = set()
active_processes: List[subprocess.Popen] = []

# ---------- Security Classes ----------

class SecurityError(Exception):
    """Raised for security-related violations"""
    pass

class ResourceExhaustionError(Exception):
    """Raised when resource limits are exceeded"""
    pass

class PipelineError(Exception):
    """Base class for pipeline-specific errors"""
    pass

# ---------- Security and Validation Functions ----------

def validate_file_path(path_str: str, must_exist: bool = True, allow_parent_dirs: bool = True) -> Path:
    """
    Validate and sanitize file paths with security checks.

    Args:
        path_str: Input path string
        must_exist: Whether file must exist
        allow_parent_dirs: Allow access to parent directories

    Returns:
        Validated Path object

    Raises:
        SecurityError: For security violations
        ValueError: For invalid paths
    """
    # Check for shell metacharacters
    if re.search(r'[;&|`$(){}[\]<>"\'*?]', path_str):
        raise SecurityError("Invalid characters detected in file path")

    # Resolve path
    try:
        path = Path(path_str).resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid path: {e}")

    # Check path length (Windows limit)
    if os.name == 'nt' and len(str(path)) > 255:
        raise ValueError("Path exceeds Windows length limit")

    # Security boundary validation
    if not allow_parent_dirs:
        # Strict mode: only allow current working directory and subdirectories
        cwd = Path.cwd().resolve()
        try:
            path.relative_to(cwd)
        except ValueError:
            raise SecurityError(f"Path outside working directory: {path}")
    else:
        # Relaxed mode: prevent access to system directories but allow user directories
        path_str_lower = str(path).lower()

        # Block dangerous system paths on Windows
        if os.name == 'nt':
            dangerous_paths = [
                'c:\\windows', 'c:\\system32', 'c:\\program files',
                'c:\\programdata', '\\windows\\', '\\system32\\',
                'c:\\users\\all users'
            ]
        else:
            # Block dangerous system paths on Unix-like systems
            dangerous_paths = [
                '/etc/', '/bin/', '/sbin/', '/usr/bin/', '/usr/sbin/',
                '/root/', '/proc/', '/sys/', '/dev/'
            ]

        for dangerous in dangerous_paths:
            if dangerous in path_str_lower:
                raise SecurityError(f"Access denied to system directory: {path}")

    # Existence checks
    if must_exist and not path.exists():
        raise ValueError(f"File does not exist: {path}")

    if must_exist and not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    return path

def validate_output_dir(dir_str: str, allow_parent_dirs: bool = True) -> Path:
    """
    Validate output directory with security checks.

    Args:
        dir_str: Output directory path
        allow_parent_dirs: Allow directories in parent paths

    Returns:
        Validated Path object
    """
    # Basic security validation
    if re.search(r'[;&|`$(){}[\]<>"\'*?]', dir_str):
        raise SecurityError("Invalid characters in output directory path")

    try:
        path = Path(dir_str).resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid output directory: {e}")

    # Security boundary validation
    if not allow_parent_dirs:
        # Strict mode: only allow current working directory and subdirectories
        cwd = Path.cwd().resolve()
        try:
            path.relative_to(cwd)
        except ValueError:
            raise SecurityError(f"Output directory outside working directory: {path}")
    else:
        # Relaxed mode: prevent access to system directories but allow user directories
        path_str_lower = str(path).lower()

        # Block dangerous system paths on Windows
        if os.name == 'nt':
            dangerous_paths = [
                'c:\\windows', 'c:\\system32', 'c:\\program files',
                'c:\\programdata', '\\windows\\', '\\system32\\',
                'c:\\users\\all users'
            ]
        else:
            # Block dangerous system paths on Unix-like systems
            dangerous_paths = [
                '/etc/', '/bin/', '/sbin/', '/usr/bin/', '/usr/sbin/',
                '/root/', '/proc/', '/sys/', '/dev/'
            ]

        for dangerous in dangerous_paths:
            if dangerous in path_str_lower:
                raise SecurityError(f"Access denied to system directory: {path}")

    return path

def validate_script_path(script_str: str) -> Path:
    """
    Validate script paths to prevent execution of malicious scripts.

    Args:
        script_str: Script path string

    Returns:
        Validated Path object
    """
    # Security validation
    if re.search(r'[;&|`$(){}[\]<>"\'*?]', script_str):
        raise SecurityError("Invalid characters in script path")

    try:
        path = Path(script_str).resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid script path: {e}")

    # Must be .py file
    if path.suffix.lower() != '.py':
        raise ValueError("Script must be a .py file")

    # Must exist
    if not path.exists() or not path.is_file():
        raise ValueError(f"Script does not exist: {path}")

    # Security check: prevent access to system directories
    path_str_lower = str(path).lower()

    if os.name == 'nt':
        dangerous_paths = [
            'c:\\windows', 'c:\\system32', 'c:\\program files',
            'c:\\programdata', '\\windows\\', '\\system32\\'
        ]
    else:
        dangerous_paths = [
            '/etc/', '/bin/', '/sbin/', '/usr/bin/', '/usr/sbin/',
            '/root/', '/proc/', '/sys/', '/dev/'
        ]

    for dangerous in dangerous_paths:
        if dangerous in path_str_lower:
            raise SecurityError(f"Script in dangerous system directory: {path}")

    return path

def validate_numeric_param(value: Union[str, int, float], param_name: str,
                         min_val: float = None, max_val: float = None,
                         return_type: str = 'str') -> Union[str, int, float]:
    """
    Validate numeric parameters with bounds checking.

    Args:
        value: Parameter value
        param_name: Parameter name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        return_type: 'str', 'int', or 'float'

    Returns:
        Validated value in requested type
    """
    try:
        numeric_value = float(value)

        # Check bounds
        if min_val is not None and numeric_value < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}")
        if max_val is not None and numeric_value > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}")

        # Return in requested type
        if return_type == 'int':
            return int(numeric_value)
        elif return_type == 'float':
            return float(numeric_value)
        else:
            return str(numeric_value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid numeric value for {param_name}: {value}")

def validate_python_executable(py_str: str = None) -> str:
    """
    Validate Python executable path.

    Args:
        py_str: Python executable path (optional)

    Returns:
        Validated Python executable path
    """
    if py_str is None:
        py_exe = sys.executable
        if not py_exe:
            # Fallback detection
            py_exe = shutil.which("python3") or shutil.which("python")
            if not py_exe:
                raise PipelineError("Cannot locate Python interpreter")
        return py_exe

    # Validate provided path
    try:
        py_path = Path(py_str).resolve()
        if not py_path.exists() or not py_path.is_file():
            raise ValueError(f"Python executable not found: {py_path}")
        return str(py_path)
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid Python executable: {e}")

def check_file_size_limits(file_path: Path, max_size_gb: float = 25.0) -> None:
    """
    Check file size to prevent DoS attacks and system crashes.

    Args:
        file_path: Path to file
        max_size_gb: Maximum allowed size in GB (reduced from 50GB to 25GB)
    """
    try:
        size_bytes = file_path.stat().st_size
        size_gb = size_bytes / (1024**3)

        if size_gb > max_size_gb:
            raise ResourceExhaustionError(
                f"File too large: {size_gb:.2f}GB (max: {max_size_gb}GB). "
                f"This size can cause system crashes. Use --max_duration to process smaller segments."
            )

        if size_gb > 5:
            logger.warning(f"Large file detected: {size_gb:.1f}GB - recommend using --max_duration for safety")

    except OSError as e:
        raise ValueError(f"Cannot access file: {e}")

def check_disk_space(path: Path, min_gb: float = 2.0) -> None:
    """
    Check available disk space before processing.

    Args:
        path: Directory path to check
        min_gb: Minimum required space in GB
    """
    try:
        free_bytes = shutil.disk_usage(path).free
        free_gb = free_bytes / (1024**3)

        if free_gb < min_gb:
            raise ResourceExhaustionError(
                f"Insufficient disk space: {free_gb:.1f}GB available, need at least {min_gb}GB"
            )
    except OSError as e:
        raise ResourceExhaustionError(f"Cannot check disk space: {e}")

def check_write_permissions(path: Path) -> None:
    """
    Check write permissions for output directory.

    Args:
        path: Directory path to check
    """
    try:
        test_file = path / f".write_test_{uuid.uuid4().hex[:8]}"
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError) as e:
        raise PermissionError(f"Cannot write to directory {path}: {e}")

def detect_wav_format(file_path: Path) -> Tuple[bool, str]:
    """
    Detect WAV file format (RIFF WAV vs RF64).

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (is_wav_format, format_type)
        format_type: 'RIFF', 'RF64', or 'UNKNOWN'
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)

        if header == b'RIFF':
            return True, 'RIFF'
        elif header == b'RF64':
            return True, 'RF64'
        else:
            return False, 'UNKNOWN'
    except (OSError, IOError) as e:
        logger.warning(f"Could not read file header: {e}")
        return False, 'UNKNOWN'

# ---------- Secure Subprocess Functions ----------

@contextmanager
def timeout_handler(seconds: int):
    """Context manager for subprocess timeouts (Unix only)."""
    if os.name != 'nt':
        def timeout_signal(signum, frame):
            raise TimeoutError(f"Process timed out after {seconds} seconds")

        old_handler = signal.signal(signal.SIGALRM, timeout_signal)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows doesn't support SIGALRM, rely on subprocess timeout
        yield

def sanitize_command_for_logging(cmd: List[str]) -> str:
    """
    Sanitize command for safe logging without information disclosure.

    Args:
        cmd: Command list

    Returns:
        Sanitized command string
    """
    sanitized = []
    for arg in cmd:
        # Hide potentially sensitive paths and data
        if any(keyword in arg.lower() for keyword in ['password', 'key', 'secret', 'token']):
            sanitized.append('[REDACTED]')
        elif len(arg) > 150:  # Very long arguments might contain sensitive data
            sanitized.append(f'{arg[:50]}...[TRUNCATED]')
        else:
            # Replace full paths with just filenames for logging
            if os.path.sep in arg and len(arg) > 50:
                sanitized.append(f'...{Path(arg).name}')
            else:
                sanitized.append(arg)
    return ' '.join(f'"{s}"' if ' ' in s else s for s in sanitized)

def secure_stream(cmd: List[str], cwd: Optional[str] = None,
                 timeout: int = 1800, max_output_mb: int = 100) -> Tuple[int, str]:
    """
    Execute subprocess with comprehensive security controls.

    Args:
        cmd: Command list (must be pre-validated)
        cwd: Working directory (optional)
        timeout: Timeout in seconds
        max_output_mb: Maximum output size in MB

    Returns:
        Tuple of (return_code, stdout_text)
    """
    global active_processes

    # Validate working directory
    if cwd:
        cwd_path = Path(cwd).resolve()
        if not cwd_path.exists() or not cwd_path.is_dir():
            raise ValueError(f"Invalid working directory: {cwd}")

    max_chars = max_output_mb * 1024 * 1024
    collected_output = []
    total_chars = 0

    try:
        # Start process with security controls
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge for simplicity but log safely
            text=True,
            bufsize=1,
            universal_newlines=True,
            shell=False,  # Critical: prevent shell injection
            # Unix: start new process group
            preexec_fn=None if os.name == 'nt' else lambda: os.setpgrp()
        )

        active_processes.append(proc)

        # Read output with size limits
        assert proc.stdout is not None
        for line in proc.stdout:
            if total_chars + len(line) > max_chars:
                logger.warning(f"Output truncated at {max_output_mb}MB limit")
                break

            collected_output.append(line)
            total_chars += len(line)
            print(line, end="")  # Stream to console

        # Wait with timeout
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.error(f"Process timed out after {timeout} seconds")
            proc.kill()
            proc.wait()  # Clean up zombie
            raise TimeoutError(f"Process exceeded {timeout} second timeout")

        return proc.returncode, "".join(collected_output)

    except Exception as e:
        if 'proc' in locals():
            try:
                proc.kill()
                proc.wait()
            except:
                pass
        raise RuntimeError(f"Subprocess execution failed: {e}")
    finally:
        if 'proc' in locals() and proc in active_processes:
            active_processes.remove(proc)

def execute_step_with_retry(cmd: List[str], step_name: str, cwd: Optional[str] = None,
                          max_retries: int = 3, timeout: int = 1800) -> Tuple[int, str]:
    """
    Execute pipeline step with retry logic.

    Args:
        cmd: Command to execute
        step_name: Step name for logging
        cwd: Working directory
        max_retries: Maximum retry attempts
        timeout: Timeout per attempt

    Returns:
        Tuple of (return_code, output)
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            logger.info(f"[{step_name}] Attempt {attempt + 1}/{max_retries}")
            rc, output = secure_stream(cmd, cwd=cwd, timeout=timeout)

            if rc == 0:
                logger.info(f"[{step_name}] Completed successfully")
                return rc, output
            else:
                last_error = f"Process failed with return code {rc}"
                logger.warning(f"[{step_name}] {last_error}")

        except Exception as e:
            last_error = str(e)
            logger.warning(f"[{step_name}] Exception: {last_error}")

        # Wait before retry (exponential backoff)
        if attempt < max_retries - 1:
            wait_time = min(30 * (2 ** attempt), 300)  # Cap at 5 minutes
            logger.info(f"[{step_name}] Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    # All retries failed
    raise PipelineError(f"{step_name} failed after {max_retries} attempts: {last_error}")

# ---------- Secure Directory and File Operations ----------

def create_secure_run_directory(base_dir: Path, wav_stem: str) -> Path:
    """
    Create run directory with proper security controls.

    Args:
        base_dir: Base output directory
        wav_stem: WAV file stem for naming

    Returns:
        Created run directory path
    """
    global cleanup_dirs

    # Create timestamp and unique identifier
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    process_id = os.getpid()
    unique_id = uuid.uuid4().hex[:8]

    # Create directory name
    dir_name = f"run_{wav_stem}_{timestamp}_{process_id}_{unique_id}"
    run_dir = base_dir / dir_name

    try:
        # Create with secure permissions
        run_dir.mkdir(parents=True, exist_ok=False)

        # Set secure permissions (owner only on Unix)
        if os.name != 'nt':
            run_dir.chmod(stat.S_IRWXU)  # 700 permissions

        # Add to cleanup registry
        cleanup_dirs.add(run_dir)

        logger.info(f"Created secure run directory: {run_dir}")
        return run_dir

    except OSError as e:
        raise PipelineError(f"Cannot create run directory: {e}")

def extract_json_robust(stdout: str, fallback_convention: str = "I+Q") -> Dict[str, Any]:
    """
    Robust JSON extraction with fallbacks and error handling.

    Args:
        stdout: Process stdout text
        fallback_convention: Fallback I/Q convention

    Returns:
        Extracted results dictionary
    """
    try:
        # Try primary JSON extraction
        first = stdout.find("{")
        last = stdout.rfind("}")
        if first != -1 and last != -1 and last > first:
            blob = stdout[first:last+1]
            results = json.loads(blob)
            if results and "best_convention" in results:
                logger.info(f"Extracted JSON successfully: {results.get('best_convention', 'unknown')}")
                return results

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed: {e}")
    except Exception as e:
        logger.warning(f"Unexpected JSON extraction error: {e}")

    # Fallback: try to parse from structured log lines
    try:
        match = re.search(r'Best.*I/Q.*convention:\s*([^\s,]+)', stdout, re.IGNORECASE)
        if match:
            convention = match.group(1)
            logger.info(f"Extracted I/Q convention from logs: {convention}")
            return {"best_convention": convention}
    except Exception as e:
        logger.warning(f"Log parsing fallback failed: {e}")

    # Final fallback
    logger.warning(f"Using fallback I/Q convention: {fallback_convention}")
    return {"best_convention": fallback_convention}

def extract_run_dir_robust(stdout: str, base_dir: Path, wav_stem: str) -> Optional[Path]:
    """
    Robust run directory extraction with multiple fallback methods.

    Args:
        stdout: Process stdout text
        base_dir: Base output directory
        wav_stem: WAV file stem

    Returns:
        Run directory path or None
    """
    # Method 1: Parse explicit output line
    try:
        match = re.search(r'Outputs written to:\s*["\']([^"\']+)["\']', stdout)
        if match:
            path = Path(match.group(1))
            if path.exists() and path.is_dir():
                logger.info(f"Found run directory from output: {path}")
                return path
    except Exception as e:
        logger.warning(f"Method 1 directory extraction failed: {e}")

    # Method 2: Find newest matching directory
    try:
        candidates = sorted(
            (d for d in base_dir.glob(f"run_{wav_stem}_*") if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True
        )
        if candidates:
            newest = candidates[0]
            logger.info(f"Found run directory by timestamp: {newest}")
            return newest
    except Exception as e:
        logger.warning(f"Method 2 directory extraction failed: {e}")

    logger.error("Could not locate run directory using any method")
    return None

def map_iq_convention_secure(step1_convention: str) -> str:
    """
    Securely map I/Q convention from Step 1 to Step 2 format.

    Args:
        step1_convention: Convention from Step 1

    Returns:
        Mapped convention for Step 2
    """
    # Secure mapping with validation
    mapping = {
        "I+jQ": "I+Q",
        "I-jQ": "I-Q",
        "Q+jI": "Q+I",
        # Handle edge cases
        "": "I+Q",
        None: "I+Q"
    }

    # Clean and validate input
    clean_conv = str(step1_convention).strip() if step1_convention else ""
    mapped = mapping.get(clean_conv)

    if mapped is None:
        logger.warning(f"Unknown I/Q convention '{clean_conv}', using I+Q fallback")
        return "I+Q"

    logger.info(f"I/Q convention mapping: '{clean_conv}' -> '{mapped}'")
    return mapped

# ---------- Resource Monitoring ----------

def validate_system_resources() -> None:
    """Validate system has adequate resources for processing."""
    try:
        import psutil

        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)

        if available_gb < 1.0:
            raise ResourceExhaustionError(f"Insufficient RAM: {available_gb:.1f}GB available")

        if available_gb < 2.0:
            logger.warning(f"Low memory condition: {available_gb:.1f}GB available")

        # Check CPU load
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            logger.warning(f"High CPU load: {cpu_percent}%")

    except ImportError:
        logger.warning("psutil not available, skipping resource validation")
    except Exception as e:
        logger.warning(f"Resource validation failed: {e}")

# ---------- Cleanup and Signal Handling ----------

def cleanup_handler() -> None:
    """Clean up resources on exit or interruption."""
    global cleanup_dirs, active_processes

    # Terminate active processes
    for proc in active_processes[:]:
        try:
            if proc.poll() is None:  # Still running
                logger.info(f"Terminating process {proc.pid}")
                proc.terminate()
                proc.wait(timeout=10)
        except Exception as e:
            logger.warning(f"Error terminating process: {e}")
            try:
                proc.kill()
            except:
                pass

    # Clean up temporary directories on failure
    # Note: Only clean up on abnormal termination, not successful completion
    signal_cleanup = hasattr(cleanup_handler, '_signal_triggered')
    if signal_cleanup:
        for dir_path in cleanup_dirs.copy():
            try:
                if dir_path.exists():
                    shutil.rmtree(dir_path, ignore_errors=True)
                    logger.info(f"Cleaned up directory: {dir_path}")
            except Exception as e:
                logger.warning(f"Cleanup failed for {dir_path}: {e}")

def signal_handler(signum: int, frame) -> None:
    """Handle interruption signals gracefully."""
    signal_name = signal.Signals(signum).name
    logger.info(f"Received signal {signal_name}, cleaning up...")

    # Mark that cleanup was triggered by signal
    cleanup_handler._signal_triggered = True
    cleanup_handler()

    sys.exit(1)

# ---------- Input Validation ----------

def validate_all_inputs(args) -> Dict[str, Any]:
    """
    Comprehensive validation of all pipeline inputs.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary of validated parameters
    """
    validated = {}

    # Determine security level
    allow_parent_dirs = not getattr(args, 'strict_paths', False)

    # Handle batch mode vs single file mode
    if len(args.wav) > 1 or args.batch_mode:
        # Batch mode - discover WAV files
        wav_files = discover_wav_files(args.wav)
        if not wav_files:
            raise ValueError("No WAV files found in specified paths")
        validated['wav_files'] = wav_files
        validated['batch_mode'] = True
        logger.info(f"Batch mode: Found {len(wav_files)} WAV files")

        # For batch mode, skip individual file validation (will be done per file)
        validated['wav_path'] = None
    else:
        # Single file mode
        validated['wav_path'] = validate_file_path(args.wav[0], must_exist=True, allow_parent_dirs=allow_parent_dirs)
        validated['batch_mode'] = False

        # Check file format and detect WAV type
        wav_path = validated['wav_path']
        if wav_path.suffix.lower() not in ['.wav', '.wave']:
            logger.warning(f"File extension '{wav_path.suffix}' is not .wav")

        # Detect WAV format (RIFF vs RF64)
        is_wav_format, wav_format = detect_wav_format(wav_path)
        if is_wav_format:
            logger.info(f"Detected {wav_format} WAV format")
            if wav_format == 'RF64':
                logger.info("RF64 format detected - optimized for large files (>4GB)")
        else:
            logger.warning(f"Unknown file format detected: {wav_format}")

        # Check file size
        check_file_size_limits(wav_path)

    # Validate output directory
    validated['out_dir'] = validate_output_dir(args.out_dir, allow_parent_dirs=allow_parent_dirs)

    # Check output directory permissions and space
    out_dir = validated['out_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
    check_write_permissions(out_dir)
    check_disk_space(out_dir)

    # Validate script paths
    script_params = ['sanity_script', 'step2_script', 'step3_script']
    for param in script_params:
        script_path = getattr(args, param)
        validated[param] = validate_script_path(script_path)

    # Determine which steps to run based on control flags
    step_control = {
        'run_step1': not args.only_step5,
        'run_step2': not args.only_step1 and not args.only_step5,
        'run_step3': not args.only_step1 and not args.only_step2 and not args.only_step5,
        'run_step3b': not args.only_step1 and not args.only_step2 and not args.only_step3 and not args.only_step5 and not args.skip_step3b,
        'run_step4': not args.only_step1 and not args.only_step2 and not args.only_step3 and not args.only_step3b and not args.only_step5 and not args.skip_step4,
        'run_step5': not args.only_step1 and not args.only_step2 and not args.only_step3 and not args.only_step3b and not args.only_step4 and not args.skip_step5
    }
    validated.update(step_control)

    # Validate script paths for enabled steps
    if step_control['run_step3b']:
        validated['step3b_script'] = validate_script_path(args.step3b_script)
    if step_control['run_step4']:
        validated['step4_script'] = validate_script_path(args.step4_script)
    if step_control['run_step5']:
        validated['step5_script'] = validate_script_path(args.step5_script)

    # Validate Python interpreter
    validated['python'] = validate_python_executable(args.python)

    # Validate numeric parameters with bounds
    # Integer parameters
    integer_validations = {
        'mem_budget_mb': (64, 32768),  # Memory limits
        'step3_mem_budget_mb': (64, 32768),
        'fft_size': (256, 65536),  # FFT size limits
        'nperseg': (256, 65536),
        'cfar_guard': (0, 100),  # Guard cells
        'cfar_train': (1, 1000),  # Training cells
        'step3_max_slices': (1, 100000),  # Slice limit
        'step3b_win': (256, 65536),  # Window size
        'step3b_min_bw': (1000, 50000000),  # Bandwidth limits
        'step3b_timeout': (60, 7200),  # Timeout range: 1 min to 2 hours
        'step4_nperseg': (256, 65536),  # FFT size for features
        'step5_batch_size': (1, 1024),  # Inference batch size
        'step5_timeout': (60, 7200),  # Timeout range: 1 min to 2 hours
    }

    for param, (min_val, max_val) in integer_validations.items():
        value = getattr(args, param)
        if value is not None:
            validated[param] = validate_numeric_param(value, param, min_val, max_val, 'int')

    # Float parameters
    float_validations = {
        'fs_hint': (1000, 1e12),  # Reasonable frequency range
        'overlap': (0.0, 1.0),  # Overlap fraction
        'prom_db': (-100, 100),  # dB range
        'cfar_k': (0.1, 100),  # CFAR multiplier
        'max_duration': (1, 86400),  # Duration in seconds (1 day max)
        'step3_rms_win_ms': (0.1, 1000),  # RMS window
        'step3_thresh_dbfs': (-120, 0),  # Threshold dBFS
        'step3_min_dur_ms': (0.1, 10000),  # Min duration
        'step3_gap_ms': (0, 1000),  # Gap merge
        'step3_pad_ms': (0, 1000),  # Padding
        'step3b_hop_frac': (0.1, 1.0),  # Hop fraction
        'step3b_oversample': (1.0, 10.0),  # Oversampling factor
        'step4_overlap': (0.0, 1.0),  # Overlap fraction for features
        'step5_seg': (1.0, 300.0),  # Chunk duration for inference
    }

    for param, (min_val, max_val) in float_validations.items():
        value = getattr(args, param)
        if value is not None:
            validated[param] = validate_numeric_param(value, param, min_val, max_val, 'float')

    # Validate boolean flags
    bool_params = ['force_full', 'adaptive_sampling', 'step3_write_audio', 'step4_emit_h5', 'step5_auto_vad']
    for param in bool_params:
        validated[param] = getattr(args, param)

    # Validate string parameters
    string_params = ['step3b_mode', 'step4_extras', 'step3b_out', 'step4_out', 'step5_out', 'step5_device']
    for param in string_params:
        validated[param] = getattr(args, param)

    # Validate optional working directories
    for param in ['sanity_path', 'step2_path', 'step3_path', 'step3b_path', 'step4_path', 'step5_path']:
        path_val = getattr(args, param)
        if path_val:
            validated[param] = str(validate_output_dir(path_val))
        else:
            validated[param] = None

    return validated

# ---------- Batch Processing Support ----------

class BatchProgress:
    """Handles batch processing progress tracking and checkpointing"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.output_dir / "batch_checkpoint.json"
        self.summary_file = self.output_dir / "batch_summary.json"

        self.completed_files = set()
        self.failed_files = set()
        self.current_file = None
        self.start_time = time.time()

        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load existing checkpoint data"""
        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)

                self.completed_files = set(data.get('completed_files', []))
                self.failed_files = set(data.get('failed_files', []))
                self.start_time = data.get('start_time', time.time())

                logger.info(f"Resuming batch: {len(self.completed_files)} completed, {len(self.failed_files)} failed")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")

    def _save_checkpoint(self):
        """Save current progress to checkpoint file"""
        try:
            data = {
                'completed_files': list(self.completed_files),
                'failed_files': list(self.failed_files),
                'current_file': str(self.current_file) if self.current_file else None,
                'start_time': self.start_time,
                'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_elapsed_hours': (time.time() - self.start_time) / 3600
            }

            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")

    def start_file(self, file_path: Path) -> bool:
        """Check if file should be processed and mark as current"""
        file_str = str(file_path.resolve())

        if file_str in self.completed_files:
            logger.info(f"File already completed: {file_path.name}")
            return False

        if file_str in self.failed_files:
            logger.info(f"Retrying previously failed file: {file_path.name}")

        self.current_file = file_path
        logger.info(f"Starting file: {file_path.name}")
        self._save_checkpoint()
        return True

    def complete_file(self, file_path: Path, success: bool):
        """Mark file as completed or failed"""
        file_str = str(file_path.resolve())

        if success:
            self.completed_files.add(file_str)
            if file_str in self.failed_files:
                self.failed_files.remove(file_str)
            logger.info(f"File completed successfully: {file_path.name}")
        else:
            self.failed_files.add(file_str)
            logger.error(f"File failed: {file_path.name}")

        self.current_file = None
        self._save_checkpoint()

        # Update summary after each file
        self._update_summary()

    def _update_summary(self):
        """Update batch summary file"""
        try:
            elapsed_hours = (time.time() - self.start_time) / 3600
            total_files = len(self.completed_files) + len(self.failed_files)

            summary = {
                'batch_start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time)),
                'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_hours': elapsed_hours,
                'completed_files': len(self.completed_files),
                'failed_files': len(self.failed_files),
                'total_processed': total_files,
                'success_rate': len(self.completed_files) / total_files if total_files > 0 else 0,
                'avg_time_per_file_hours': elapsed_hours / total_files if total_files > 0 else 0,
                'completed_list': list(self.completed_files),
                'failed_list': list(self.failed_files)
            }

            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not update summary: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current batch status"""
        elapsed_hours = (time.time() - self.start_time) / 3600
        total_processed = len(self.completed_files) + len(self.failed_files)

        return {
            'completed': len(self.completed_files),
            'failed': len(self.failed_files),
            'total_processed': total_processed,
            'elapsed_hours': elapsed_hours,
            'success_rate': len(self.completed_files) / total_processed if total_processed > 0 else 0
        }

def discover_wav_files(input_paths: List[str]) -> List[Path]:
    """Discover all WAV files from input paths (files or directories)"""
    wav_files = []

    for input_path in input_paths:
        path_obj = Path(input_path)

        if path_obj.is_file():
            # Single file
            if path_obj.suffix.lower() in ['.wav', '.wave']:
                wav_files.append(path_obj)
            else:
                logger.warning(f"Skipping non-WAV file: {path_obj}")

        elif path_obj.is_dir():
            # Directory - find all WAV files
            found_files = []
            extensions = ['*.wav', '*.WAV', '*.wave', '*.WAVE']

            for ext in extensions:
                found_files.extend(path_obj.glob(ext))

            logger.info(f"Found {len(found_files)} WAV files in directory: {path_obj}")
            wav_files.extend(found_files)

        else:
            logger.error(f"Invalid path (not file or directory): {path_obj}")

    # Remove duplicates and sort
    wav_files = list(set(wav_files))
    wav_files.sort(key=lambda x: x.name.lower())

    logger.info(f"Total WAV files discovered: {len(wav_files)}")

    # Show first 10 files with sizes
    for i, wav_file in enumerate(wav_files[:10], 1):
        try:
            size_gb = wav_file.stat().st_size / (1024**3)
            logger.info(f"  {i:2d}. {wav_file.name} ({size_gb:.1f}GB)")
        except:
            logger.info(f"  {i:2d}. {wav_file.name} (size unknown)")

    if len(wav_files) > 10:
        logger.info(f"  ... and {len(wav_files) - 10} more files")

    return wav_files

def cleanup_memory():
    """Force memory cleanup"""
    gc.collect()
    try:
        # Additional cleanup for numpy arrays
        import numpy as np
        if hasattr(np, '_cleanup'):
            np._cleanup()
    except:
        pass

def check_memory_usage() -> Tuple[float, float]:
    """Check current memory usage in GB"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        used_gb = (memory.total - memory.available) / (1024**3)
        available_gb = memory.available / (1024**3)
        return used_gb, available_gb
    except ImportError:
        logger.warning("psutil not available - cannot monitor memory")
        return 0.0, 8.0  # fallback values

def wait_for_memory(required_gb: float = 4.0, timeout_minutes: int = 10) -> bool:
    """Wait for sufficient memory to be available"""
    timeout_seconds = timeout_minutes * 60
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        used_gb, available_gb = check_memory_usage()

        if available_gb >= required_gb:
            logger.info(f"Memory OK: {available_gb:.1f}GB available (need {required_gb:.1f}GB)")
            return True

        logger.warning(f"Low memory: {available_gb:.1f}GB available, need {required_gb:.1f}GB")
        logger.info("Forcing memory cleanup and waiting...")

        cleanup_memory()
        time.sleep(30)  # Wait 30 seconds before checking again

    logger.error(f"Timeout waiting for memory after {timeout_minutes} minutes")
    return False

def create_results_index(output_dir: Path, batch_results: dict):
    """Create an index file for easy viewing of batch results"""
    try:
        index_file = output_dir / "batch_results_index.html"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Batch Processing Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .success {{ color: green; }}
        .failed {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>RF Pipeline Batch Processing Results</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Files:</strong> {batch_results['total_files']}</p>
        <p><strong class="success">Completed:</strong> {len(batch_results['completed'])}</p>
        <p><strong class="failed">Failed:</strong> {len(batch_results['failed'])}</p>
        <p><strong>Success Rate:</strong> {len(batch_results['completed']) / batch_results['total_files'] * 100:.1f}%</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <h2>File Details</h2>
    <table>
        <tr>
            <th>File</th>
            <th>Status</th>
            <th>Results Directory</th>
        </tr>
"""

        # Add completed files
        for file_path in batch_results['completed']:
            file_name = Path(file_path).name
            html_content += f"""
        <tr>
            <td>{file_name}</td>
            <td class="success">✅ Completed</td>
            <td>Check output directory for timestamped folder</td>
        </tr>"""

        # Add failed files
        for file_path in batch_results['failed']:
            file_name = Path(file_path).name
            html_content += f"""
        <tr>
            <td>{file_name}</td>
            <td class="failed">❌ Failed</td>
            <td>Check logs for error details</td>
        </tr>"""

        html_content += """
    </table>
</body>
</html>"""

        with open(index_file, 'w') as f:
            f.write(html_content)

        logger.info(f"Results index created: {index_file}")

    except Exception as e:
        logger.warning(f"Could not create results index: {e}")

# ---------- Command Building ----------

def build_secure_command(py_exe: str, script_path: Path, args_dict: Dict[str, Any]) -> List[str]:
    """
    Build subprocess command with proper validation and escaping.

    Args:
        py_exe: Python executable path
        script_path: Script file path
        args_dict: Argument dictionary

    Returns:
        Command list ready for subprocess execution
    """
    cmd = [py_exe, "-u", str(script_path)]

    # Handle positional arguments first (wav file for most scripts)
    if 'wav' in args_dict:
        cmd.append(str(args_dict['wav']))

    # Add named arguments in secure manner
    for key, value in args_dict.items():
        if key == 'wav':
            continue  # Already handled as positional

        if value is not None and value != "":
            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

    return cmd

# ---------- Main Pipeline Function ----------

def process_single_file(wav_path: Path, validated: dict, args) -> dict:
    """
    Process a single WAV file through the entire pipeline.

    Args:
        wav_path: Path to WAV file
        validated: Validated parameters
        args: Command line arguments

    Returns:
        Dictionary with processing results
    """
    py_exe = validated['python']
    out_base = validated['out_dir']

    logger.info(f"Processing WAV file: {wav_path}")
    logger.info(f"Output directory: {out_base}")

    # Create run directory based on WAV filename and timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_{wav_path.stem}"
    run_dir = out_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Add run directory to cleanup list
    cleanup_dirs.add(run_dir)

    results = {}

    # --- Step 1: Sanity Check ---
    logger.info("=== Step 1: Sanity Check ===")

    step1_args = {
        'wav': str(wav_path),
        'out_dir': str(run_dir),
        'mem_budget_mb': max(64, validated.get('mem_budget_mb', 512))
    }

    if validated.get('fs_hint'):
        step1_args['fs_hint'] = validated['fs_hint']
    if validated.get('fft_size'):
        step1_args['fft_size'] = validated['fft_size']
    if validated.get('chunk_mode') and validated.get('start_time') is not None:
        step1_args['start_time'] = validated['start_time']
        if validated.get('end_time') is not None:
            step1_args['end_time'] = validated['end_time']

    step1_cmd = build_secure_command(py_exe, validated['sanity_script'], step1_args)

    logger.info(f"Command: {sanitize_command_for_logging(step1_cmd)}")

    rc1, out1 = execute_step_with_retry(
        step1_cmd, "Step-1",
        cwd=validated.get('sanity_path'),
        timeout=args.timeout,
        max_retries=args.max_retries
    )

    # Parse Step-1 results
    results1 = {}
    try:
        # Try multiple possible Step-1 output files
        step1_candidates = [
            run_dir / "summary.json",
            run_dir / f"sanity_check_{wav_path.stem}_stats.json"
        ]

        # Also look for stats files in subdirectories
        for subdir in run_dir.iterdir():
            if subdir.is_dir():
                step1_candidates.append(subdir / f"sanity_check_{wav_path.stem}_stats.json")

        step1_loaded = False
        for candidate in step1_candidates:
            if candidate.exists():
                with open(candidate, 'r') as f:
                    results1 = json.load(f)
                logger.info(f"Loaded Step-1 results from {candidate.name}: {len(results1)} items")
                step1_loaded = True
                break

        if not step1_loaded:
            logger.warning("Step-1 summary.json not found")
    except Exception as e:
        logger.warning(f"Could not parse Step-1 results: {e}")

    results['step1'] = results1

    if not validated['run_step2']:
        logger.info("Step 2 and beyond skipped per configuration")
        return results

    # Parse Step-1 results for I/Q convention mapping
    best_conv_step1 = results1.get("best_convention", "I+Q")
    mapped_conv = map_iq_convention_secure(best_conv_step1)
    logger.info(f"I/Q convention: {best_conv_step1} -> {mapped_conv}")

    # Only continue if Step 2 is enabled
    if validated['run_step2']:
        # --- Step 2: Wideband exploration ---
        logger.info("=== Step 2: Wideband Exploration ===")

        step2_args = {
            'wav': str(wav_path),
            'out': str(run_dir),
            'nperseg': validated.get('nperseg', 4096),
            'overlap': validated.get('overlap', 0.5),
            'prom_db': validated.get('prom_db', 8.0),
            'cfar_k': validated.get('cfar_k', 3.0),
            'cfar_guard': validated.get('cfar_guard', 1),
            'cfar_train': validated.get('cfar_train', 6),
            'conv': mapped_conv
        }

        if validated.get('fs_hint'):
            step2_args['fs_hint'] = validated['fs_hint']

        # Apply industrial mode optimizations for massive files (22GB+)
        if args.industrial_mode:
            logger.info("🏭 INDUSTRIAL MODE: Optimizing for massive files (22GB+)")
            file_duration_s = results1.get('file_duration_s', 300.0)
            # Ultra-conservative settings for 22GB files
            step2_args['max_duration'] = min(args.chunk_duration, 5.0)  # Never more than 5s chunks
            step2_args['nperseg'] = 512   # Smallest FFT size
            step2_args['cfar_k'] = 4.0    # Very strict detection to reduce processing
            step2_args['prom_db'] = 10.0  # Only strong peaks
            step2_args['force_full'] = True  # Override safety checks
            logger.info(f"🏭 Industrial mode: max_duration={step2_args['max_duration']}s, nperseg=512, ULTRA-CONSERVATIVE")

        # Apply hybrid mode optimizations
        elif args.hybrid_mode:
            logger.info("Hybrid mode enabled: applying memory optimizations")
            # Set optimized parameters for memory efficiency
            step2_args['max_duration'] = min(10.0, results1.get('file_duration_s', 60.0))
            step2_args['nperseg'] = 1024  # Smaller FFT for less memory
            step2_args['cfar_k'] = 2.5    # Less strict detection
            step2_args['prom_db'] = 5.0   # Less strict peaks
            logger.info(f"Hybrid mode: max_duration={step2_args['max_duration']}s, nperseg=1024")

        # For batch mode, automatically set max_duration for memory safety on large files
        logger.info(f"Batch mode check: validated.get('batch_mode')={validated.get('batch_mode')}")
        if validated.get('batch_mode') and not args.hybrid_mode:
            # Get file duration from Step 1 results if available
            file_duration_s = results1.get('file_duration_s', 0)
            logger.info(f"File duration from Step 1: {file_duration_s}s")
            if file_duration_s > 3.0:  # If file is longer than 3 seconds
                # Use a very conservative max_duration to prevent memory issues (aim for < 2GB memory usage)
                safe_duration = min(3.0, file_duration_s)
                step2_args['max_duration'] = safe_duration
                logger.info(f"Batch mode: Setting max_duration={safe_duration}s for memory safety (file duration: {file_duration_s:.1f}s)")
            elif validated.get('max_duration') and args.max_duration != 60.0:
                step2_args['max_duration'] = validated['max_duration']
        elif validated.get('max_duration') and args.max_duration != 60.0 and not args.hybrid_mode:
            step2_args['max_duration'] = validated['max_duration']
        if validated.get('force_full'):
            step2_args['force_full'] = True
        if validated.get('adaptive_sampling'):
            step2_args['adaptive_sampling'] = True

        step2_cmd = build_secure_command(py_exe, validated['step2_script'], step2_args)
        logger.info(f"Command: {sanitize_command_for_logging(step2_cmd)}")

        rc2, out2 = execute_step_with_retry(
            step2_cmd, "Step-2",
            cwd=validated.get('step2_path'),
            timeout=args.timeout,
            max_retries=args.max_retries
        )

    # Only run Step 3B if enabled (Signal slicing for ML)
    if validated['run_step3b']:
        # Check for carriers CSV (required for Step 3B)
        carriers_csv = run_dir / "carriers.csv"
        if not carriers_csv.is_file():
            logger.warning("No carriers.csv found - Step 3B requires Step 2 carrier detection")
            logger.info("Skipping Step 3B due to missing carriers.csv")
        else:
            # --- Step 3B: ML Signal Slicing (creates HDF5) ---
            logger.info("=== Step 3B: ML Signal Slicing (HDF5) ===")

            step3b_out_dir = validated['out_dir'] / args.step3b_out
            step3b_out_dir.mkdir(parents=True, exist_ok=True)

            step3b_args = {
                'wav': str(wav_path),
                'out': str(step3b_out_dir),
                'mem_budget_mb': 512,
                'rms_win_ms': 5.0,
                'thresh_dbfs': -60.0,
                'min_dur_ms': 5.0,
                'gap_ms': 3.0,
                'pad_ms': 2.0,
                'write_audio': True,
                'max_slices': 1000
            }

            if validated.get('fs_hint'):
                step3b_args['fs_hint'] = validated['fs_hint']

            step3b_cmd = build_secure_command(py_exe, validated['step3b_script'], step3b_args)
            logger.info(f"Command: {sanitize_command_for_logging(step3b_cmd)}")

            step3b_timeout = validated.get('step3b_timeout', 600)
            logger.info(f"Step 3B timeout set to {step3b_timeout} seconds")

            rc3b, out3b = execute_step_with_retry(
                step3b_cmd, "Step-3B",
                cwd=validated.get('step3b_path'),
                timeout=step3b_timeout,
                max_retries=args.max_retries
            )

    # Only run Step 4 if enabled (Feature extraction)
    if validated['run_step4']:
        step3b_out_dir = validated['out_dir'] / args.step3b_out
        slices_h5 = step3b_out_dir / "slices.h5"
        meta_json = step3b_out_dir / "meta.json"

        if not (slices_h5.is_file() and meta_json.is_file()):
            logger.warning("Step 4 requires Step 3B outputs (slices.h5 and meta.json)")
            logger.info("Skipping Step 4 due to missing Step 3B outputs")
        else:
            # --- Step 4: Feature Extraction ---
            logger.info("=== Step 4: Feature Extraction ===")

            step4_out_dir = validated['out_dir'] / args.step4_out
            step4_out_dir.mkdir(parents=True, exist_ok=True)

            step4_args = {
                'slices_h5': str(slices_h5),
                'meta_json': str(meta_json),
                'out_dir': str(step4_out_dir),
                'nperseg': validated.get('step4_nperseg', 2048),
                'overlap': args.step4_overlap,
                'extras': args.step4_extras
            }

            if args.step4_emit_h5:
                step4_args['emit_engineered_h5'] = True

            step4_cmd = build_secure_command(py_exe, validated['step4_script'], step4_args)
            logger.info(f"Command: {sanitize_command_for_logging(step4_cmd)}")

            rc4, out4 = execute_step_with_retry(
                step4_cmd, "Step-4",
                cwd=validated.get('step4_path'),
                timeout=args.timeout,
                max_retries=args.max_retries
            )

    # Only run Step 5 if enabled (Inference)
    if validated['run_step5']:
        step5_out_dir = run_dir / args.step5_out
        step5_out_dir.mkdir(parents=True, exist_ok=True)

        # --- Step 5: Inference ---
        logger.info("=== Step 5: Inference ===")

        step5_args = {
            'in': str(wav_path),
            'model_ts': args.step5_model,
            'labels_json': args.step5_labels,
            'out_dir': str(step5_out_dir),
            'device': args.step5_device,
            'seg': 2.0 if args.hybrid_mode else args.step5_seg,  # Smaller chunks in hybrid mode
            'batch_size': 32 if args.hybrid_mode else validated.get('step5_batch_size', 64)  # Smaller batches
        }

        # Apply industrial mode optimizations for Step 5 (massive files)
        if args.industrial_mode:
            step5_args['seg'] = 1.0  # Ultra-small 1s chunks
            step5_args['batch_size'] = 8  # Ultra-small batches
            step5_args['vad_abs_dbfs'] = -25.0  # High threshold, less processing
            step5_args['psd_max_frames'] = 1024  # Minimal PSD processing
            step5_args['psd_preview_sr'] = 500000  # Aggressive downsampling
            step5_args['cuda_gc'] = True
            step5_args['auto_batch'] = True
            step5_args['min_batch'] = 2  # Extreme fallback
            step5_args['limit'] = 60  # Limit to first 60 chunks for 22GB files (test first)
            logger.info("🏭 Industrial mode: Step 5 ULTRA-CONSERVATIVE (1s chunks, batch=8, limited chunks)")

        # Apply hybrid mode optimizations for Step 5
        elif args.hybrid_mode:
            step5_args['vad_abs_dbfs'] = -30.0  # Skip expensive auto-VAD
            step5_args['psd_max_frames'] = 4096  # Reduce PSD processing
            step5_args['psd_preview_sr'] = 1000000  # More aggressive downsampling
            step5_args['cuda_gc'] = True  # Enable GPU memory cleanup
            step5_args['auto_batch'] = True  # Enable dynamic batch sizing
            step5_args['min_batch'] = 8  # Minimum batch size for OOM recovery
            logger.info("Hybrid mode: Step 5 optimizations applied (2s chunks, smaller batches)")

        if validated.get('fs_hint'):
            step5_args['sr'] = int(validated['fs_hint'])

        if args.step5_auto_vad:
            step5_args['auto_vad'] = True

        # Set channel order based on Step 1 results
        try:
            if best_conv_step1 == "Q+jI":
                step5_args['channel_order'] = 'QI'
            elif best_conv_step1 in ["I+jQ", "I-jQ"]:
                step5_args['channel_order'] = 'IQ'
            else:
                step5_args['channel_order'] = 'IQ'  # default
        except:
            step5_args['channel_order'] = 'IQ'  # fallback

        step5_cmd = build_secure_command(py_exe, validated['step5_script'], step5_args)
        logger.info(f"Command: {sanitize_command_for_logging(step5_cmd)}")

        step5_timeout = validated.get('step5_timeout', 1200)
        logger.info(f"Step 5 timeout set to {step5_timeout} seconds")

        rc5, out5 = execute_step_with_retry(
            step5_cmd, "Step-5",
            cwd=validated.get('step5_path'),
            timeout=step5_timeout,
            max_retries=args.max_retries
        )

    # Success - don't clean up successful runs
    if run_dir in cleanup_dirs:
        cleanup_dirs.remove(run_dir)

    logger.info("=== Single File Pipeline Completed Successfully ===")
    logger.info(f"All outputs in: {run_dir}")

    return results


def process_batch(wav_files: List[Path], validated: dict, args) -> dict:
    """
    Process multiple WAV files sequentially with checkpointing.

    Args:
        wav_files: List of WAV file paths
        validated: Validated parameters
        args: Command line arguments

    Returns:
        Dictionary with batch processing results
    """
    logger.info(f"Starting batch processing of {len(wav_files)} files")

    # Initialize batch progress tracking
    out_base = validated['out_dir']
    batch_progress = BatchProgress(out_base)

    # Progress info is already loaded in constructor
    if args.resume and (batch_progress.completed_files or batch_progress.failed_files):
        logger.info(f"Resuming: {len(batch_progress.completed_files)} completed, {len(batch_progress.failed_files)} failed")

    batch_results = {
        'total_files': len(wav_files),
        'completed': [],
        'failed': [],
        'results': {}
    }

    start_time = time.time()

    for idx, wav_file in enumerate(wav_files, 1):
        wav_str = str(wav_file)

        # Skip if already completed
        if wav_str in batch_progress.completed_files:
            logger.info(f"[{idx}/{len(wav_files)}] Skipping completed file: {wav_file.name}")
            batch_results['completed'].append(wav_str)
            continue

        logger.info(f"[{idx}/{len(wav_files)}] Processing: {wav_file.name}")

        # Memory check before processing
        used_gb, available_gb = check_memory_usage()
        if available_gb < args.batch_memory_limit_gb:
            logger.warning(f"Low memory: {available_gb:.1f}GB available, waiting...")
            wait_for_memory(args.batch_memory_limit_gb)

        try:
            # Process single file
            file_results = process_single_file(wav_file, validated, args)

            # Mark as completed
            batch_progress.complete_file(wav_file, success=True)
            batch_results['completed'].append(wav_str)
            batch_results['results'][wav_str] = file_results

            logger.info(f"[{idx}/{len(wav_files)}] Completed: {wav_file.name}")

        except Exception as e:
            logger.error(f"[{idx}/{len(wav_files)}] Failed: {wav_file.name} - {e}")
            batch_progress.complete_file(wav_file, success=False)
            batch_results['failed'].append(wav_str)

        # Memory cleanup between files
        cleanup_memory()

        # Progress reporting
        elapsed = time.time() - start_time
        if idx > 0:
            avg_time = elapsed / idx
            remaining = len(wav_files) - idx
            eta = avg_time * remaining
            logger.info(f"Progress: {idx}/{len(wav_files)}, ETA: {eta/3600:.1f}h ({eta/60:.0f}min)")

    # Final results
    total_time = time.time() - start_time
    success_rate = len(batch_results['completed']) / len(wav_files)

    logger.info(f"Batch processing completed:")
    logger.info(f"  Total files: {len(wav_files)}")
    logger.info(f"  Completed: {len(batch_results['completed'])}")
    logger.info(f"  Failed: {len(batch_results['failed'])}")
    logger.info(f"  Success rate: {success_rate*100:.1f}%")
    logger.info(f"  Total time: {total_time/3600:.1f}h ({total_time/60:.0f}min)")

    # Save batch summary
    if args.batch_results_index:
        create_results_index(out_base, batch_results)

    return batch_results


def main():
    """Main pipeline execution with comprehensive error handling."""

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_handler)

    # Argument parsing
    ap = argparse.ArgumentParser(
        description="Secure I/Q pipeline runner (Step-1 sanity -> Step-2 analysis -> Step-3 slicing).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument("wav", nargs='+', help="Path(s) to I/Q .wav file(s) or directories containing WAV files")
    ap.add_argument("--fs_hint", type=float, default=None,
                   help="Override WAV header Fs (Hz)")
    ap.add_argument("--out_dir", default="out_report",
                   help="Base output folder")

    # Step-1 options
    ap.add_argument("--sanity_script", default="1_sanity_check_iq.py",
                   help="Path to Step-1 script")
    ap.add_argument("--mem_budget_mb", type=int, default=512,
                   help="RAM budget per chunk for Step-1")
    ap.add_argument("--fft_size", type=int, default=None,
                   help="FFT size per chunk for Step-1 (power of 2)")
    ap.add_argument("--sanity_path", default=None,
                   help="Working dir for Step-1 (optional)")

    # Step-2 options
    ap.add_argument("--step2_script", default="2_wideband_exploration.py",
                   help="Path to Step-2 script")
    ap.add_argument("--nperseg", type=int, default=4096,
                   help="FFT window length")
    ap.add_argument("--overlap", type=float, default=0.5,
                   help="STFT/Welch overlap fraction")
    ap.add_argument("--prom_db", type=float, default=8.0,
                   help="Min prominence (dB) for PSD peaks")
    ap.add_argument("--cfar_k", type=float, default=3.0,
                   help="CFAR k (higher=stricter)")
    ap.add_argument("--cfar_guard", type=int, default=1,
                   help="CFAR guard cells")
    ap.add_argument("--cfar_train", type=int, default=6,
                   help="CFAR training cells")
    ap.add_argument("--max_duration", type=float, default=60.0,
                   help="Max processing duration for Step-2 sampling")
    ap.add_argument("--start_time", type=float, default=0.0,
                   help="Start time (seconds) within the file")
    ap.add_argument("--end_time", type=float, default=None,
                   help="End time (seconds) within the file (None for end of file)")
    ap.add_argument("--chunk_mode", action="store_true",
                   help="Enable chunk processing mode (process only specified time segment)")
    ap.add_argument("--force_full", action="store_true",
                   help="Force full Step-2 processing (no sampling)")
    ap.add_argument("--adaptive_sampling", action="store_true",
                   help="Use adaptive sampling strategy to optimize IRR in Step-2")
    ap.add_argument("--step2_path", default=None,
                   help="Working dir for Step-2 (optional)")

    # Step-3 options
    ap.add_argument("--step3_script", default="3_signal_detection_slicing.py",
                   help="Path to Step-3 script")
    ap.add_argument("--step3_mem_budget_mb", type=int, default=512,
                   help="RAM budget per chunk for Step-3")
    ap.add_argument("--step3_rms_win_ms", type=float, default=5.0,
                   help="RMS window (ms) for Step-3")
    ap.add_argument("--step3_thresh_dbfs", type=float, default=-60.0,
                   help="Threshold (dBFS) for Step-3")
    ap.add_argument("--step3_min_dur_ms", type=float, default=5.0,
                   help="Min duration (ms) for Step-3 slices")
    ap.add_argument("--step3_gap_ms", type=float, default=3.0,
                   help="Merge gaps shorter than this (ms)")
    ap.add_argument("--step3_pad_ms", type=float, default=2.0,
                   help="Padding (ms) for Step-3 slices")
    ap.add_argument("--step3_write_audio", action="store_true",
                   help="Write individual WAV slices in Step-3")
    ap.add_argument("--step3_max_slices", type=int, default=1000,
                   help="Limit number of slices")
    ap.add_argument("--step3_path", default=None,
                   help="Working dir for Step-3 (optional)")

    # Step-3B options (Signal slicing for ML - creates HDF5)
    ap.add_argument("--step3b_script", default="3_signal_detection_slicing.py",
                   help="Path to Step-3B slicing script")
    ap.add_argument("--step3b_mode", default="carriers", choices=["carriers", "bursts"],
                   help="Slicing mode: carriers (whole file) or bursts (activity-based)")
    ap.add_argument("--step3b_win", type=int, default=16384,
                   help="Window size for slicing")
    ap.add_argument("--step3b_hop_frac", type=float, default=0.5,
                   help="Hop fraction for windowing")
    ap.add_argument("--step3b_oversample", type=float, default=4.0,
                   help="Oversampling factor")
    ap.add_argument("--step3b_min_bw", type=int, default=100000,
                   help="Minimum bandwidth (Hz)")
    ap.add_argument("--step3b_out", default="slices_out",
                   help="Output directory for Step-3B slices")
    ap.add_argument("--step3b_path", default=None,
                   help="Working dir for Step-3B (optional)")
    ap.add_argument("--step3b_timeout", type=int, default=600,
                   help="Timeout for Step-3B in seconds (default: 600)")

    # Step-4 options (Feature extraction)
    ap.add_argument("--step4_script", default="4_feature_extraction.py",
                   help="Path to Step-4 feature extraction script")
    ap.add_argument("--step4_nperseg", type=int, default=2048,
                   help="FFT window length for feature extraction")
    ap.add_argument("--step4_overlap", type=float, default=0.5,
                   help="Overlap fraction for feature extraction")
    ap.add_argument("--step4_extras", default="amp,phase,dfreq,cosphi,sinphi,d_amp,d2phase,cum40,cum41,cum42",
                   help="Engineered channels for feature extraction")
    ap.add_argument("--step4_emit_h5", action="store_true",
                   help="Emit engineered HDF5 file for deep learning")
    ap.add_argument("--step4_out", default="step4_out",
                   help="Output directory for Step-4 features")
    ap.add_argument("--step4_path", default=None,
                   help="Working dir for Step-4 (optional)")

    # Step-5 options (Inference)
    ap.add_argument("--step5_script", default="5_inference.py",
                   help="Path to Step-5 inference script")
    ap.add_argument("--step5_model", default="Model/best_model_script_cuda.pt",
                   help="Path to TorchScript model file")
    ap.add_argument("--step5_labels", default="Model/selected_labels.json",
                   help="Path to labels JSON file")
    ap.add_argument("--step5_device", default="cuda", choices=["cuda", "cpu"],
                   help="Device for inference (cuda/cpu)")
    ap.add_argument("--step5_seg", type=float, default=10.0,
                   help="Chunk duration in seconds for inference")
    ap.add_argument("--step5_batch_size", type=int, default=64,
                   help="Batch size for inference")
    ap.add_argument("--step5_auto_vad", action="store_true",
                   help="Enable automatic VAD for inference")
    ap.add_argument("--step5_out", default="step5_out",
                   help="Output directory for Step-5 inference results")
    ap.add_argument("--step5_path", default=None,
                   help="Working dir for Step-5 (optional)")
    ap.add_argument("--step5_timeout", type=int, default=1200,
                   help="Timeout for Step-5 in seconds (default: 1200)")

    # Step control options
    ap.add_argument("--only_step1", action="store_true",
                   help="Run only Step 1 (sanity check)")
    ap.add_argument("--only_step2", action="store_true",
                   help="Run only Steps 1-2 (sanity check + wideband exploration)")
    ap.add_argument("--only_step3", action="store_true",
                   help="Run only Steps 1-3 (up to basic signal detection)")
    ap.add_argument("--only_step3b", action="store_true",
                   help="Run only Steps 1-3B (up to ML signal slicing)")
    ap.add_argument("--only_step4", action="store_true",
                   help="Run only Steps 1-4 (up to feature extraction)")
    ap.add_argument("--only_step5", action="store_true",
                   help="Run only Step 5 (inference only - requires existing data)")
    ap.add_argument("--skip_step3b", action="store_true",
                   help="Skip Step 3B (ML signal slicing)")
    ap.add_argument("--skip_step4", action="store_true",
                   help="Skip Step 4 (feature extraction)")
    ap.add_argument("--skip_step5", action="store_true",
                   help="Skip Step 5 (inference)")

    # General options
    ap.add_argument("--python", default=None,
                   help="Path to Python interpreter (default: current)")
    ap.add_argument("--timeout", type=int, default=1800,
                   help="Timeout per step in seconds")
    ap.add_argument("--max_retries", type=int, default=3,
                   help="Maximum retry attempts per step")
    ap.add_argument("--strict_paths", action="store_true",
                   help="Use strict path validation (only current dir and subdirs)")

    # Batch processing options
    ap.add_argument("--batch_mode", action="store_true",
                   help="Enable batch processing mode for multiple files")
    ap.add_argument("--resume", action="store_true",
                   help="Resume interrupted batch processing")
    ap.add_argument("--batch_memory_limit_gb", type=float, default=6.0,
                   help="Memory limit (GB) for batch processing safety")
    ap.add_argument("--batch_results_index", action="store_true",
                   help="Create results index file for easy viewing")

    # Memory optimization options
    ap.add_argument("--hybrid_mode", action="store_true",
                   help="Enable hybrid processing mode (5-channel enrichment + optimizations)")
    ap.add_argument("--skip_plots", action="store_true",
                   help="Skip all visualization/plotting for memory savings")
    ap.add_argument("--industrial_mode", action="store_true",
                   help="Industrial mode for massive files (22GB+, 5min+) with extreme memory safety")
    ap.add_argument("--chunk_duration", type=float, default=10.0,
                   help="Processing chunk duration in seconds for large files")
    ap.add_argument("--memory_limit_gb", type=float, default=8.0,
                   help="Maximum memory usage per processing step (GB)")

    try:
        args = ap.parse_args()

        # System resource validation
        logger.info("Validating system resources...")
        validate_system_resources()

        # Comprehensive input validation
        logger.info("Validating inputs...")
        validated = validate_all_inputs(args)

        out_base = validated['out_dir']
        py_exe = validated['python']

        # Handle batch mode vs single file mode
        if validated['batch_mode']:
            logger.info(f"Batch processing mode: {len(validated['wav_files'])} files")
            process_batch(validated['wav_files'], validated, args)
        else:
            wav_path = validated['wav_path']
            logger.info(f"Single file processing: {wav_path}")
            logger.info(f"Output directory: {out_base}")
            process_single_file(wav_path, validated, args)

        logger.info("=== All Processing Completed Successfully ===")

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
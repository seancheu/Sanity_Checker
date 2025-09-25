# Batch Processing Guide for Large RF64 Files

This guide covers processing multiple large RF64 WAV files (22GB+ each) safely and efficiently.

## Quick Start

### For a folder with 50+ files of 22GB each:

```bash
# Process entire folder with 30-second chunks
python batch_processor.py /path/to/folder/with/wav/files --fs_hint 20000000 --chunk_duration 30

# More conservative 10-second chunks
python batch_processor.py /path/to/folder/with/wav/files --fs_hint 20000000 --chunk_duration 10
```

### For specific files:

```bash
# Process specific files
python batch_processor.py file1.wav file2.wav file3.wav --fs_hint 20000000 --chunk_duration 60

# Resume interrupted batch processing
python batch_processor.py /path/to/folder --fs_hint 20000000 --resume
```

## Key Features

### ✅ **Memory Safety**
- **Chunked processing**: Breaks each 22GB file into manageable time segments
- **Sequential file processing**: One file at a time to prevent memory exhaustion
- **Automatic memory monitoring**: Waits for memory availability between operations
- **Memory cleanup**: Force garbage collection between files and chunks

### ✅ **Resumability**
- **Progress tracking**: Saves progress to `batch_progress.json`
- **Resume capability**: Skip already-processed files with `--resume`
- **Error isolation**: One failed file doesn't stop the entire batch

### ✅ **Safety Controls**
- **Memory limits**: Configurable memory thresholds (default: 8GB)
- **Timeout protection**: Per-chunk timeouts prevent hangs
- **Error handling**: Robust error handling with detailed logging

## Processing Strategy

### **File-Level (Sequential)**
```
File 1 (22GB) → File 2 (22GB) → File 3 (22GB) → ... → File 50 (22GB)
```

### **Within Each File (Chunked)**
```
File 1: [0-30s] → [30-60s] → [60-90s] → ... → [300s-end]
```

### **Memory Management**
- Clean up after each chunk
- Wait for memory availability before starting next chunk
- Monitor system memory usage continuously
- Force garbage collection between files

## Configuration Options

### **Chunk Duration**
```bash
--chunk_duration 10   # 10-second chunks (very conservative)
--chunk_duration 30   # 30-second chunks (recommended)
--chunk_duration 60   # 60-second chunks (faster but more memory)
```

### **Memory Limits**
```bash
--memory_limit_gb 4   # Conservative (4GB limit)
--memory_limit_gb 8   # Standard (8GB limit)
--memory_limit_gb 16  # High-memory systems (16GB limit)
```

### **Output Organization**
```
batch_output/
├── batch_progress.json          # Progress tracking
├── batch_summary.json           # Final statistics
├── file1_stem/                  # Results for file1.wav
│   ├── chunk_0000/              # First chunk (0-30s)
│   ├── chunk_0001/              # Second chunk (30-60s)
│   └── ...
├── file2_stem/                  # Results for file2.wav
└── ...
```

## Memory Requirements

### **Minimum System Requirements**
- **RAM**: 8GB+ available (16GB+ recommended for 50 files)
- **Storage**: ~2GB free space per processed chunk
- **CPU**: Any modern multi-core processor

### **Processing Time Estimates**
- **Per 30s chunk**: ~2-5 minutes
- **Per 22GB file**: ~2-4 hours (depends on file content)
- **50 files batch**: ~4-8 days (can run unattended)

## Example Workflows

### **Conservative Processing** (Lowest memory usage)
```bash
python batch_processor.py /data/rf_files --fs_hint 20000000 \
    --chunk_duration 10 --memory_limit_gb 4 \
    --output_dir conservative_output
```

### **Balanced Processing** (Recommended)
```bash
python batch_processor.py /data/rf_files --fs_hint 20000000 \
    --chunk_duration 30 --memory_limit_gb 8 \
    --output_dir balanced_output
```

### **Fast Processing** (High-memory systems)
```bash
python batch_processor.py /data/rf_files --fs_hint 20000000 \
    --chunk_duration 60 --memory_limit_gb 16 \
    --output_dir fast_output
```

## Monitoring Progress

### **Real-time Monitoring**
```bash
# Monitor log file
tail -f batch_processing.log

# Check progress file
cat batch_output/batch_progress.json
```

### **Progress Information**
- Files completed vs. total files
- Current chunk progress within file
- Estimated time remaining
- Memory usage statistics
- Success/failure rates

## Error Recovery

### **Common Issues & Solutions**

**Memory exhaustion:**
```bash
# Reduce chunk duration
--chunk_duration 10

# Lower memory limit (forces more aggressive cleanup)
--memory_limit_gb 4
```

**Process hangs:**
```bash
# Reduce timeout per chunk
# Edit batch_processor.py line ~200: timeout=1800  # 30 minutes
```

**Partial failures:**
```bash
# Resume processing to retry failed files
python batch_processor.py /data/rf_files --fs_hint 20000000 --resume
```

## Advanced Usage

### **Single File Chunking** (Alternative to batch processor)
```bash
# Process just one file in chunks using the regular pipeline
python run_iq_pipeline_v2.py huge_file.wav --fs_hint 20000000 \
    --chunk_mode --start_time 0 --end_time 30

python run_iq_pipeline_v2.py huge_file.wav --fs_hint 20000000 \
    --chunk_mode --start_time 30 --end_time 60
```

### **Custom Processing Steps**
```bash
# Skip Step 3B (most memory-intensive) for batch processing
# This is done automatically in batch_processor.py
python run_iq_pipeline_v2.py file.wav --fs_hint 20000000 --skip_step3b
```

## Safety Recommendations

1. **Test First**: Run on 1-2 files before processing entire folder
2. **Monitor Resources**: Keep Task Manager/htop open to watch memory
3. **Stable System**: Don't run other heavy applications during processing
4. **Backup Important**: Keep original files safe - processing creates copies
5. **Resume Capability**: Use `--resume` if processing is interrupted

**The batch processor is designed to handle your 50+ files of 22GB each safely without system crashes.**
# NLVR2 Evaluation Guide

## Overview

`evaluate_nlvr2.py` evaluates PROVE on the NLVR2 benchmark with robust checkpointing and automatic metrics computation.

## Quick Start

### 1. Download Images (if not already done)

Images must be downloaded separately. For balanced_test1:

```bash
python data/nlvr/nlvr2/util/download_images.py \
    data/nlvr/nlvr2/data/balanced/balanced_test1.json \
    data/nlvr/nlvr2/images \
    data/nlvr/nlvr2/util/hashes/test1_hashes.json
```

**Important Notes**:
- Download can take several hours (~5% of images may be inaccessible from original URLs)
- **Hash verification**: Script automatically verifies downloaded images using perceptual hashing
- **Corrupted images**: Automatically deleted if hash doesn't match expected value
- **Failed downloads**: Logged to `*_failed_imgs.txt` and `*_failed_hashes.txt`

After download, only valid, hash-verified images remain on disk.

### 2. Run Quick Test (10 examples)

```bash
python evaluate_nlvr2.py --split balanced_test1 --num_examples 10
```

### 3. Run Full Evaluation

**Recommended: Balanced Test Set** (2,316 examples, 50/50 True/False)
```bash
python evaluate_nlvr2.py --split balanced_test1
```

**Full Test Set** (6,967 examples)
```bash
python evaluate_nlvr2.py --split test1
```

## Available Splits

| Split | Examples | Description | Recommended |
|-------|----------|-------------|-------------|
| `balanced_test1` | 2,316 | Balanced subset (50/50 True/False) | ✅ **START HERE** |
| `test1` | 6,967 | Full public test set | Optional |
| `balanced_dev` | 2,300 | Balanced dev set | For tuning |
| `dev` | 6,982 | Full dev set | For tuning |
| `test2` | 6,970 | Hidden test (now public) | Not needed |

## Command-Line Options

```bash
python evaluate_nlvr2.py \
    --split balanced_test1 \         # Which split to evaluate
    --num_examples 100 \              # Limit examples (default: all)
    --resume \                        # Resume from checkpoint
    --checkpoint_freq 10 \            # Save every N examples (default: 10)
    --output_dir nlvr2_results \      # Output directory
    --verbose                         # Show PROVE pipeline output
```

## Checkpointing

### Automatic Checkpointing
- Saves every 10 examples by default (configurable with `--checkpoint_freq`)
- Checkpoint file: `nlvr2_checkpoint.json` (in project root)
- Atomic writes prevent corruption

### Resuming

If evaluation is interrupted, simply add `--resume`:

```bash
python evaluate_nlvr2.py --split balanced_test1 --resume
```

The script will automatically continue from the last checkpoint.

### Manual Checkpoint Management

```bash
# View checkpoint status
cat nlvr2_checkpoint.json | jq '.completed, .total_examples'

# Clear checkpoint (start fresh)
rm nlvr2_checkpoint.json
```

## Output Files

All outputs saved to `nlvr2_results/` (or specified `--output_dir`):

```
nlvr2_results/
├── predictions_balanced_test1.csv     # Official CSV format (identifier,prediction)
├── detailed_results_balanced_test1.json  # Full results with timing & ground truth
├── analysis_balanced_test1.txt        # Analysis report with confusion matrix
└── (checkpoint in project root)       # nlvr2_checkpoint.json
```

## Metrics

The script automatically runs the official NLVR2 metrics script and reports:

1. **Accuracy**: Percentage of correct predictions
2. **Consistency**: For each unique sentence, checks if ALL associated image pairs are predicted correctly

Example output:
```
accuracy=0.5234
consistency=0.4891
```

## Expected Runtime

| Split | Examples | Time (est.) |
|-------|----------|-------------|
| balanced_test1 | 2,316 | ~3-4 hours |
| test1 | 6,967 | ~10-12 hours |

Based on ~5-6 seconds per example (varies by question complexity).

## Error Handling

- **Missing images**: Automatically skips with warning (images failed download or hash verification)
- **Corrupted images**: Download script deletes them automatically; evaluation validates remaining with PIL
- **PROVE failures**: Logs error, continues with next example
- **Partial completions**: Checkpoint preserves all completed examples

Failed examples are logged in `detailed_results_*.json` under `failed_examples`.

### Image Download & Validation Pipeline

**During Download** (`download_images.py`):
1. Download image from URL
2. Compute perceptual hash using `imagehash.average_hash()`
3. Compare with expected hash from `util/hashes/*.json`
4. If hash mismatches → **delete file** and log to `*_failed_hashes.txt`
5. If download fails → log to `*_failed_imgs.txt`

**Result**: Only valid, hash-verified images remain on disk.

**During Evaluation** (`evaluate_nlvr2.py`):
1. Check if both images in pair exist (missing = deleted during download)
2. Validate remaining images with PIL (catch any edge cases)
3. Only process examples with valid image pairs
4. Log any remaining issues to `corrupted_images.json`

**Failed Download Logs**:
- `*_failed_imgs.txt`: HTTP errors, timeouts, connection issues
- `*_failed_hashes.txt`: Downloaded but hash verification failed (corrupted)
- `*_checked_imgs.txt`: Successfully processed URLs

These files allow you to identify and retry failed downloads if needed.

## Analysis Report

The analysis report (`analysis_*.txt`) includes:

- Overall accuracy
- True/False breakdown
- Confusion matrix
- Timing statistics
- Failed example summary

Example:
```
================================================================================
NLVR2 EVALUATION ANALYSIS
================================================================================

Total examples processed: 2300
Failed examples: 16
Correct predictions: 1204
Accuracy: 0.5234 (52.34%)

True examples: 1150 (acc: 0.5391)
False examples: 1150 (acc: 0.5078)

Average time per example: 5.2s
Total time: 3.3h (200m)

Confusion Matrix:
              Predicted
              True  False
Actual True    620    530
       False   566    584
================================================================================
```

## Tips

1. **Start small**: Test with `--num_examples 10` first to verify setup
2. **Use balanced sets**: More reliable evaluation with 50/50 label distribution
3. **Monitor progress**: Script shows real-time accuracy and ETA
4. **Check failed examples**: Review `detailed_results_*.json` for errors
5. **Save checkpoints frequently**: Use `--checkpoint_freq 5` for safety

## Troubleshooting

### "Missing images" warning
→ Download images using the command shown in output

### "Checkpoint split doesn't match"
→ Clear checkpoint: `rm nlvr2_checkpoint.json`

### PROVE timeout/OOM
→ Restart with `--resume` to continue

### Low accuracy
→ Check `analysis_*.txt` for confusion matrix and error patterns

## Submitting Results

The leaderboard is no longer actively maintained (as of Aug 2022), but for reference:

1. Use predictions CSV: `nlvr2_results/predictions_test1.csv`
2. Format: `identifier,prediction` (e.g., "test1-0-1-0,True")
3. Evaluate with: `python data/nlvr/nlvr2/eval/metrics.py predictions.csv data.json`

## Citation

If using NLVR2, cite:

```bibtex
@inproceedings{suhr2019corpus,
  title={A corpus for reasoning about natural language grounded in photographs},
  author={Suhr, Alane and Zhou, Stephanie and Zhang, Ally and Zhang, Iris and Bai, Huajun and Artzi, Yoav},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages={6418--6428},
  year={2019}
}
```

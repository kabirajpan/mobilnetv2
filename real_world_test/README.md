# Real-World Image Testing

This folder contains scripts for testing real-world apple leaf images with trained models.

## Files

- `real_world_test.py` - Main testing script
- `test_images/` - Place your test images here (optional)

## Usage

### Simple Output (Just the Result)

```bash
# Simple output: Just shows RUST, HEALTHY, ALTERNARIA, or SCAB
python real_world_test.py --model v2 --image path/to/image.jpg --simple

# Example output:
# ✅ RESULT: RUST
#    Confidence: 95.2%
```

### Test Single Image (Detailed)

```bash
# Test with MobileNetV2
python real_world_test.py --model v2 --image path/to/image.jpg

# Test with MobileNetV3
python real_world_test.py --model v3 --image path/to/image.jpg

# Compare both models
python real_world_test.py --model both --image path/to/image.jpg
```

### Test Multiple Images (Folder)

```bash
# Test all images in a folder with MobileNetV2
python real_world_test.py --model v2 --folder path/to/folder/

# Test all images in a folder with MobileNetV3
python real_world_test.py --model v3 --folder path/to/folder/
```

### Options

- `--model`: Choose model (`v2`, `v3`, or `both` for comparison)
- `--image`: Path to single image file
- `--folder`: Path to folder containing images
- `--simple`: **Simple output mode** - Just shows result (RUST, HEALTHY, etc.) without details
- `--no-plot`: Don't show plots (useful for batch processing)
- `--save-plot`: Save visualization plots to `test_results/` folder

## Examples

### Simple Mode (Recommended for Quick Testing)

```bash
# Simple output - just tells you RUST, HEALTHY, ALTERNARIA, or SCAB
python real_world_test.py --model v2 --image test.jpg --simple

# Test folder with simple output
python real_world_test.py --model v2 --folder test_images/ --simple
```

### Detailed Mode

```bash
# Test single image and save plot
python real_world_test.py --model v2 --image test_images/apple_leaf.jpg --save-plot

# Test folder without showing plots
python real_world_test.py --model v3 --folder test_images/ --no-plot

# Compare both models on same image
python real_world_test.py --model both --image test_images/apple_leaf.jpg
```

## Output

The script will:
1. Load the trained model
2. Preprocess the image
3. Make predictions
4. Show class probabilities
5. Display visualization (unless `--no-plot` is used)
6. Save results to `test_results/` folder (if `--save-plot` is used)

## Supported Image Formats

- JPG/JPEG
- PNG
- BMP

## Requirements

- Trained models must exist in:
  - `../mobilenetv2/checkpoints/`
  - `../mobilenetv3/checkpoints/`
- If exact model name not found, script will use the most recent `.keras` file


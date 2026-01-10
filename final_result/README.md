# Final Results - Model Comparison

This folder contains scripts and results for comparing all trained models.

## Files

- `compare_all_models.py` - Main comparison script that analyzes all models

## Usage

```bash
cd final_result
python compare_all_models.py
```

## What It Does

The comparison script:

1. **Loads Results** from all model folders:
   - mobilenetv2
   - mobilenetv3
   - resnet
   - efficientnet

2. **Compares Metrics**:
   - Validation Accuracy
   - TTA Accuracy
   - MAP (Mean Average Precision)
   - Macro/Weighted Precision, Recall, F1-Score
   - Generalization Gap
   - Training Time

3. **Generates Visualizations**:
   - Main comparison plot (4 subplots)
   - MAP comparison plot (with per-class AP)
   - Detailed metrics comparison
   - Model ranking

4. **Creates Reports**:
   - CSV comparison table
   - Text report with analysis
   - High-resolution plots (300 DPI)

## Outputs

- **comparison_results/**: 
  - `model_comparison.csv` - Comparison table
  - `comparison_report_[timestamp].txt` - Detailed text report

- **comparison_plots/**:
  - `model_comparison_main.png` - Main comparison visualization
  - `map_comparison.png` - MAP comparison with per-class AP
  - `detailed_metrics_comparison.png` - Detailed metrics breakdown

## Requirements

- All models must be trained first
- Results JSON files should exist in each model's `results/` folder
- The script will automatically find the most recent results for each model

## Best Model Selection

The script ranks models based on:
- **Overall Score** = 40% TTA Accuracy + 40% MAP Score + 20% Macro F1-Score

The best model is highlighted in the output and reports.


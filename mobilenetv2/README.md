# MobileNetV2 Training

This folder contains the MobileNetV2 model training scripts for apple disease classification.

## Files

- `train.py` - Optimized training script targeting 95%+ accuracy with comprehensive evaluation
- `improved_surgical_boost.py` - Previous improved version with bug fixes

## Training

```bash
cd mobilenetv2
python train.py
```

## Expected Results

- **Standard Accuracy**: ~93-94%
- **TTA Accuracy**: 95%+
- **MAP (Mean Average Precision)**: 95%+
- **Training Time**: ~5-6 hours

## Model Architecture

- **Base**: MobileNetV2 (ImageNet pre-trained)
- **Input Size**: 160×160×3
- **Classifier Head**: 
  - GlobalAveragePooling2D
  - BatchNormalization + Dense(320) + ReLU + Dropout(0.5)
  - BatchNormalization + Dense(160) + ReLU + Dropout(0.4)
  - Dense(4) + Softmax

## Key Features

- ✅ Mixed precision training (30% faster)
- ✅ Two-phase training (transfer learning + fine-tuning)
- ✅ Optimized TTA with 7 augmentations
- ✅ Cosine annealing with warmup
- ✅ Enhanced augmentation strategy
- ✅ **MAP (Mean Average Precision) calculation**
- ✅ **Comprehensive evaluation metrics**
- ✅ **Visualization plots for research paper**

## Evaluation Metrics

The training script calculates and saves:

1. **Accuracy Metrics**
   - Training Accuracy
   - Validation Accuracy
   - TTA (Test-Time Augmentation) Accuracy

2. **MAP (Mean Average Precision)**
   - Per-class Average Precision (AP)
   - Overall MAP score

3. **Per-Class Metrics**
   - Precision, Recall, F1-Score per class
   - Macro and Weighted averages

4. **ROC & PR Curves**
   - ROC curves with AUC for each class
   - Precision-Recall curves with AP for each class

## Visualization Plots

The script generates two comprehensive plot files:

1. **evaluation_plots_[timestamp].png**
   - Confusion Matrix (raw and normalized)
   - ROC Curves for all classes
   - Precision-Recall Curves for all classes

2. **training_curves_[timestamp].png**
   - Training/Validation Accuracy over epochs
   - Training/Validation Loss over epochs
   - Per-class metrics bar chart
   - Performance summary text

## Outputs

- **checkpoints/**: Saved model files (.keras)
  - `best_mobilenetv2_optimized_phase1.keras`
  - `best_mobilenetv2_optimized_phase2.keras`
  - `best_mobilenetv2_optimized.keras` (final)

- **logs/**: Training logs (CSV)
  - `phase1_[timestamp].csv`
  - `phase2_[timestamp].csv`

- **results/**: Evaluation results
  - `results_[timestamp].json` - Complete metrics in JSON format
  - `evaluation_plots_[timestamp].png` - Evaluation visualizations
  - `training_curves_[timestamp].png` - Training progress plots

## Research Paper Ready

All outputs are formatted for research paper inclusion:
- High-resolution plots (300 DPI)
- Comprehensive metrics (MAP, Precision, Recall, F1)
- Per-class performance analysis
- Training curves and evaluation visualizations


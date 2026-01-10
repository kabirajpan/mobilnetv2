"""
OPTIMIZED ResNet50 Training for Apple Disease Classification
==============================================================
Mirrors MobileNetV2 training strategy for fair comparison:
1. ✅ Mixed precision training (30% faster)
2. ✅ Fixed TTA with proper normalization
3. ✅ Enhanced augmentation strategy
4. ✅ Same hyperparameters as MobileNetV2
5. ✅ Better learning rate scheduling
6. ✅ Optimized layer unfreezing strategy
7. ✅ Comprehensive evaluation with MAP

Target: Compare with MobileNetV2 (90% baseline)
Expected time: ~2-2.5 hours (i3 6th gen)
"""

import os
import sys
import time
import math
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

# Add parent directory to path for scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ============================
# CONFIGURATION
# ============================
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.set_soft_device_placement(True)

# Enable mixed precision for faster training
try:
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)
    print("🚀 Mixed precision enabled: float16 compute, float32 variables")
except:
    print("⚠️  Mixed precision not available, using float32")

# SAME hyperparameters as MobileNetV2 for fair comparison
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 12
EPOCHS_PHASE1 = 12
EPOCHS_PHASE2 = 6
DATA_DIR = "../apple_dataset/raw"
AUGMENTED_DATA_DIR = "../apple_dataset/augmented"
MODEL_SAVE_PATH = "checkpoints/best_resnet50_optimized.keras"
BEST_MODEL_PATH = "checkpoints/best_resnet50_optimized_phase2.keras"
LOG_DIR = "./logs"
RESULTS_DIR = "./results"
LABEL_SMOOTHING = 0.08

print("=" * 70)
print("🎯 OPTIMIZED ResNet50 Training - Research Paper Ready")
print("=" * 70)
print(f"📐 Image size: {IMAGE_SIZE}")
print(f"📦 Batch size: {BATCH_SIZE}")
print(f"🎓 Label smoothing: {LABEL_SMOOTHING}")
print(f"⚡ Mixed precision: Enabled")
print(f"📊 Phase 1 epochs: {EPOCHS_PHASE1} (transfer learning)")
print(f"📊 Phase 2 epochs: {EPOCHS_PHASE2} (fine-tuning)")
print(f"📊 Total epochs: {EPOCHS_PHASE1 + EPOCHS_PHASE2}")
print(f"⏱️  Expected time: ~2-2.5 hours (i3 6th gen)")
print("=" * 70)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# ============================
# DATA SOURCE SELECTION
# ============================
def select_data_source():
    """Select best available data source"""
    if os.path.exists(AUGMENTED_DATA_DIR):
        aug_classes = [
            d
            for d in os.listdir(AUGMENTED_DATA_DIR)
            if os.path.isdir(os.path.join(AUGMENTED_DATA_DIR, d))
        ]

        if len(aug_classes) >= 4:
            total_aug = 0
            for class_dir in aug_classes:
                class_path = os.path.join(AUGMENTED_DATA_DIR, class_dir)
                images = [
                    f
                    for f in os.listdir(class_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                total_aug += len(images)

            if total_aug > 500:
                print(f"✅ Using pre-augmented data: {total_aug:,} images")
                return AUGMENTED_DATA_DIR, True

    print(f"✅ Using original data from: {DATA_DIR}")
    return DATA_DIR, False


selected_data_dir, using_preaugmented = select_data_source()

# ============================
# SAME DATA GENERATORS AS MOBILENETV2
# ============================
if using_preaugmented:
    print("🔄 Enhanced augmentation strategy for pre-augmented data")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=12,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.08,
        brightness_range=[0.92, 1.08],
        shear_range=0.05,
        fill_mode='reflect',
        validation_split=0.2,
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255, validation_split=0.2
    )
else:
    print("🔄 Heavy augmentation for original data")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="reflect",
        validation_split=0.2,
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255, validation_split=0.2
    )

# Create generators
train_generator = train_datagen.flow_from_directory(
    selected_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    interpolation='bilinear',
)

val_generator = val_datagen.flow_from_directory(
    selected_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    interpolation='bilinear',
)

print(f"\n📊 Dataset Statistics:")
print(f"   Training samples: {train_generator.samples:,}")
print(f"   Validation samples: {val_generator.samples:,}")
print(f"   Classes: {list(train_generator.class_indices.keys())}")

# ============================
# RESNET50 MODEL ARCHITECTURE
# ============================
def create_resnet_model(num_classes):
    """Create ResNet50 model with custom head (similar to MobileNetV2)"""
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # Enhanced classifier head (same structure as MobileNetV2)
    x = Dense(320, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(160, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    
    # Output layer with float32 for mixed precision compatibility
    predictions = Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


model, base_model = create_resnet_model(train_generator.num_classes)

print(f"\n🧠 ResNet50 Model: {model.count_params():,} parameters")
print(f"   (MobileNetV2 has ~3.5M parameters for comparison)")

# ============================
# CALLBACKS
# ============================
def create_callbacks(phase_name):
    """Create callbacks with same settings as MobileNetV2"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return [
        ModelCheckpoint(
            BEST_MODEL_PATH.replace(".keras", f"_{phase_name}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        CSVLogger(os.path.join(LOG_DIR, f"{phase_name}_{timestamp}.csv")),
    ]


# ============================
# PHASE 1: TRANSFER LEARNING
# ============================
print("\n" + "=" * 70)
print("PHASE 1: TRANSFER LEARNING (Frozen Backbone)")
print("=" * 70)

base_model.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"],
)

phase1_start = time.time()
print("🚀 Starting Phase 1 training...")

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE1,
    callbacks=create_callbacks("phase1"),
    verbose=1,
)

phase1_time = time.time() - phase1_start
print(f"\n✅ Phase 1 completed in {phase1_time / 60:.1f} minutes")

# ============================
# PHASE 2: FINE-TUNING
# ============================
print("\n" + "=" * 70)
print("PHASE 2: FINE-TUNING (Optimized Strategy)")
print("=" * 70)

base_model.trainable = True

# Unfreeze last 40 layers of ResNet50 (similar ratio to MobileNetV2)
# ResNet50 has 175 layers total, freeze first 135
for layer in base_model.layers[:135]:
    layer.trainable = False

for layer in base_model.layers[135:]:
    layer.trainable = True

trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"🔓 Trainable parameters in Phase 2: {trainable:,}")


# Same cosine annealing with warmup as MobileNetV2
def cosine_annealing_with_warmup(epoch):
    """Cosine annealing with warmup for better convergence"""
    warmup_epochs = 4
    max_lr = 2.5e-5
    min_lr = 5e-7

    if epoch < warmup_epochs:
        return min_lr + (max_lr - min_lr) * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (EPOCHS_PHASE2 - warmup_epochs)
        return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2


model.compile(
    optimizer=Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"],
)

callbacks_phase2 = create_callbacks("phase2")
callbacks_phase2.append(LearningRateScheduler(cosine_annealing_with_warmup, verbose=1))

print("🚀 Starting Phase 2 training with optimized strategy...")
phase2_start = time.time()

history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks_phase2,
    verbose=1,
)

phase2_time = time.time() - phase2_start
total_time = phase1_time + phase2_time

print(f"\n✅ Phase 2 completed in {phase2_time / 60:.1f} minutes")
print(f"⏱️  Total training time: {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)")

# ============================
# EVALUATION WITH TTA
# ============================
print("\n" + "=" * 70)
print("EVALUATION WITH OPTIMIZED TTA")
print("=" * 70)

# Load best model
try:
    best_model = tf.keras.models.load_model(
        BEST_MODEL_PATH.replace(".keras", "_phase2.keras")
    )
    print("✅ Loaded best Phase 2 model")
except:
    best_model = model
    print("ℹ️  Using current model")

# Standard evaluation
val_loss, val_acc = best_model.evaluate(val_generator, verbose=0)
train_loss, train_acc = best_model.evaluate(train_generator, verbose=0)

print(f"\n📊 Standard Evaluation:")
print(f"   Training Accuracy:   {train_acc:.4f} ({train_acc * 100:.2f}%)")
print(f"   Validation Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")
print(f"   Generalization Gap:  {abs(train_acc - val_acc):.4f}")


# Same TTA function as MobileNetV2
def optimized_tta(model, val_gen, n_aug=7):
    """Optimized TTA with proper normalization (same as MobileNetV2)"""
    print(f"\n🔄 Running Optimized TTA ({n_aug} augmentations)...")
    start = time.time()

    # Baseline predictions
    val_gen.reset()
    baseline = model.predict(val_gen, verbose=0)
    all_preds = [baseline]

    # Load RAW images
    raw_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.2 if using_preaugmented else 0.2
    )

    raw_gen = raw_datagen.flow_from_directory(
        selected_data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    # Load all raw images
    x_raw = []
    y_all = []
    for i in range(len(raw_gen)):
        batch_x, batch_y = raw_gen[i]
        x_raw.append(batch_x)
        y_all.append(batch_y)

    x_raw = np.concatenate(x_raw)[: len(baseline)]
    y_all = np.concatenate(y_all)[: len(baseline)]

    # Enhanced TTA datagen
    tta_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=18,
        width_shift_range=0.12,
        height_shift_range=0.12,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.1,
        brightness_range=[0.88, 1.12],
        shear_range=0.08,
    )

    # Apply augmentations
    for aug_idx in range(n_aug):
        print(f"   TTA {aug_idx + 1}/{n_aug}...", end="\r")

        aug_preds = []
        for start_idx in range(0, len(x_raw), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(x_raw))
            batch = x_raw[start_idx:end_idx]

            # Augment RAW images
            flow = tta_gen.flow(batch, batch_size=len(batch), shuffle=False)
            aug_batch = next(flow)

            # CRITICAL: Normalize AFTER augmentation
            aug_norm = aug_batch / 255.0

            # Predict
            preds = model.predict(aug_norm, verbose=0)
            aug_preds.extend(preds)

        all_preds.append(np.array(aug_preds))

    # Ensemble
    final_preds = np.mean(all_preds, axis=0)
    pred_classes = np.argmax(final_preds, axis=1)
    true_classes = np.argmax(y_all, axis=1)

    tta_acc = np.mean(pred_classes == true_classes)
    elapsed = time.time() - start

    print(f"\n   ✅ TTA complete in {elapsed / 60:.1f} minutes")
    return tta_acc


# Run TTA
tta_acc = optimized_tta(best_model, val_generator, n_aug=7)
improvement = tta_acc - val_acc

# ============================
# COMPREHENSIVE EVALUATION
# ============================
print("\n" + "=" * 70)
print("COMPREHENSIVE EVALUATION WITH MAP & METRICS")
print("=" * 70)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Get predictions
val_generator.reset()
y_pred_proba = best_model.predict(val_generator, verbose=0)
y_pred_classes = np.argmax(y_pred_proba, axis=1)

# Get true labels
val_generator.reset()
y_true = []
for i in range(len(val_generator)):
    _, batch_y = val_generator[i]
    y_true.extend(np.argmax(batch_y, axis=1))
    if len(y_true) >= len(y_pred_classes):
        break

y_true = np.array(y_true[:len(y_pred_classes)])

# Class names
class_names = list(val_generator.class_indices.keys())
num_classes = len(class_names)

# Calculate MAP
print("\n📊 Calculating MAP (Mean Average Precision)...")
ap_scores = []
for i in range(num_classes):
    y_true_binary = (y_true == i).astype(int)
    y_pred_binary = y_pred_proba[:, i]
    ap = average_precision_score(y_true_binary, y_pred_binary)
    ap_scores.append(ap)
    print(f"   {class_names[i]}: AP = {ap:.4f}")

map_score = np.mean(ap_scores)
print(f"\n🎯 MAP (Mean Average Precision): {map_score:.4f} ({map_score * 100:.2f}%)")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names, digits=4))

# Per-class metrics
precision_per_class = precision_score(y_true, y_pred_classes, average=None)
recall_per_class = recall_score(y_true, y_pred_classes, average=None)
f1_per_class = f1_score(y_true, y_pred_classes, average=None)

# Macro averages
macro_precision = np.mean(precision_per_class)
macro_recall = np.mean(recall_per_class)
macro_f1 = np.mean(f1_per_class)

# Weighted averages
weighted_precision = precision_score(y_true, y_pred_classes, average='weighted')
weighted_recall = recall_score(y_true, y_pred_classes, average='weighted')
weighted_f1 = f1_score(y_true, y_pred_classes, average='weighted')

print(f"\n📈 Overall Metrics:")
print(f"   Macro Precision: {macro_precision:.4f}")
print(f"   Macro Recall:    {macro_recall:.4f}")
print(f"   Macro F1-Score:  {macro_f1:.4f}")
print(f"   Weighted Precision: {weighted_precision:.4f}")
print(f"   Weighted Recall:    {weighted_recall:.4f}")
print(f"   Weighted F1-Score:  {weighted_f1:.4f}")

# ============================
# CREATE VISUALIZATION PLOTS
# ============================
print("\n📊 Creating visualization plots...")

# 1. Confusion Matrix and ROC Curves
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0, 0], cbar_kws={'label': 'Count'})
axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Predicted', fontsize=12)
axes[0, 0].set_ylabel('Actual', fontsize=12)

# Normalized Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0, 1], cbar_kws={'label': 'Proportion'})
axes[0, 1].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Predicted', fontsize=12)
axes[0, 1].set_ylabel('Actual', fontsize=12)

# ROC Curves
axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
roc_aucs = []
for i in range(num_classes):
    y_true_binary = (y_true == i).astype(int)
    y_pred_binary = y_pred_proba[:, i]
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
    roc_auc = auc(fpr, tpr)
    roc_aucs.append(roc_auc)
    axes[1, 0].plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

axes[1, 0].set_xlabel('False Positive Rate', fontsize=12)
axes[1, 0].set_ylabel('True Positive Rate', fontsize=12)
axes[1, 0].set_title('ROC Curves', fontsize=14, fontweight='bold')
axes[1, 0].legend(loc='lower right')
axes[1, 0].grid(True, alpha=0.3)

# Precision-Recall Curves
for i in range(num_classes):
    y_true_binary = (y_true == i).astype(int)
    y_pred_binary = y_pred_proba[:, i]
    precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
    ap = average_precision_score(y_true_binary, y_pred_binary)
    axes[1, 1].plot(recall, precision, label=f'{class_names[i]} (AP = {ap:.3f})')

axes[1, 1].set_xlabel('Recall', fontsize=12)
axes[1, 1].set_ylabel('Precision', fontsize=12)
axes[1, 1].set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
axes[1, 1].legend(loc='lower left')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plots_file = os.path.join(RESULTS_DIR, f"evaluation_plots_{timestamp}.png")
plt.savefig(plots_file, dpi=300, bbox_inches='tight')
print(f"✅ Evaluation plots saved: {plots_file}")
plt.close()

# 2. Training Curves
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Combine histories
combined_acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
combined_val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
combined_loss = history_phase1.history['loss'] + history_phase2.history['loss']
combined_val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']

epochs = range(1, len(combined_acc) + 1)
phase1_epochs = len(history_phase1.history['accuracy'])

# Accuracy plot
axes[0, 0].plot(epochs, combined_acc, 'b-o', label='Training Accuracy', linewidth=2, markersize=4)
axes[0, 0].plot(epochs, combined_val_acc, 'r-o', label='Validation Accuracy', linewidth=2, markersize=4)
axes[0, 0].axvline(x=phase1_epochs, color='green', linestyle='--', alpha=0.7, label='Fine-tuning Start')
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 1])

# Loss plot
axes[0, 1].plot(epochs, combined_loss, 'b-o', label='Training Loss', linewidth=2, markersize=4)
axes[0, 1].plot(epochs, combined_val_loss, 'r-o', label='Validation Loss', linewidth=2, markersize=4)
axes[0, 1].axvline(x=phase1_epochs, color='green', linestyle='--', alpha=0.7, label='Fine-tuning Start')
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epochs')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Per-class metrics bar chart
x_pos = np.arange(len(class_names))
width = 0.25

axes[1, 0].bar(x_pos - width, precision_per_class, width, label='Precision', alpha=0.8)
axes[1, 0].bar(x_pos, recall_per_class, width, label='Recall', alpha=0.8)
axes[1, 0].bar(x_pos + width, f1_per_class, width, label='F1-Score', alpha=0.8)
axes[1, 0].set_xlabel('Classes', fontsize=12)
axes[1, 0].set_ylabel('Score', fontsize=12)
axes[1, 0].set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
axes[1, 0].set_ylim([0, 1.1])
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Summary metrics
summary_text = f"""Model Performance Summary

Overall Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)
TTA Accuracy: {tta_acc:.4f} ({tta_acc*100:.2f}%)

MAP: {map_score:.4f} ({map_score*100:.2f}%)

Macro Precision: {macro_precision:.4f}
Macro Recall: {macro_recall:.4f}
Macro F1-Score: {macro_f1:.4f}

Weighted Precision: {weighted_precision:.4f}
Weighted Recall: {weighted_recall:.4f}
Weighted F1-Score: {weighted_f1:.4f}

Training Time: {total_time/60:.1f} min ({total_time/3600:.2f} hrs)
"""

axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11,
                verticalalignment='center', transform=axes[1, 1].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
axes[1, 1].axis('off')

plt.tight_layout()
training_plots_file = os.path.join(RESULTS_DIR, f"training_curves_{timestamp}.png")
plt.savefig(training_plots_file, dpi=300, bbox_inches='tight')
print(f"✅ Training curves saved: {training_plots_file}")
plt.close()

# ============================
# SAVE RESULTS
# ============================
results = {
    "model": "ResNet50",
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs_phase1": EPOCHS_PHASE1,
    "epochs_phase2": EPOCHS_PHASE2,
    "label_smoothing": LABEL_SMOOTHING,
    "training_samples": int(train_generator.samples),
    "validation_samples": int(val_generator.samples),
    "classes": class_names,
    "train_accuracy": float(train_acc),
    "val_accuracy": float(val_acc),
    "tta_accuracy": float(tta_acc),
    "tta_improvement": float(improvement),
    "generalization_gap": float(abs(train_acc - val_acc)),
    "map_score": float(map_score),
    "ap_per_class": {class_names[i]: float(ap_scores[i]) for i in range(num_classes)},
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_precision": float(weighted_precision),
    "weighted_recall": float(weighted_recall),
    "weighted_f1": float(weighted_f1),
    "precision_per_class": {class_names[i]: float(precision_per_class[i]) for i in range(num_classes)},
    "recall_per_class": {class_names[i]: float(recall_per_class[i]) for i in range(num_classes)},
    "f1_per_class": {class_names[i]: float(f1_per_class[i]) for i in range(num_classes)},
    "roc_auc_per_class": {class_names[i]: float(roc_aucs[i]) for i in range(num_classes)},
    "phase1_time_minutes": float(phase1_time / 60),
    "phase2_time_minutes": float(phase2_time / 60),
    "total_time_minutes": float(total_time / 60),
    "total_time_hours": float(total_time / 3600),
    "using_preaugmented": using_preaugmented,
    "timestamp": timestamp
}

results_file = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# ============================
# FINAL RESULTS
# ============================
print("\n" + "=" * 70)
print("🎉 OPTIMIZED TRAINING RESULTS")
print("=" * 70)
print(f"📊 Standard Accuracy:  {val_acc:.4f} ({val_acc * 100:.2f}%)")
print(f"🚀 TTA Accuracy:       {tta_acc:.4f} ({tta_acc * 100:.2f}%)")
print(f"📈 TTA Improvement:    {improvement:+.4f} ({improvement * 100:+.2f}%)")
print(f"🎯 MAP Score:          {map_score:.4f} ({map_score * 100:.2f}%)")
print(f"📊 Macro F1-Score:     {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
print(f"⏱️  Total Time:        {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)")
print("=" * 70)

if tta_acc >= 0.96:
    print("🏆 OUTSTANDING: 96%+ accuracy achieved!")
elif tta_acc >= 0.95:
    print("🏆 SUCCESS: 95%+ accuracy achieved! 🎯")
elif tta_acc >= 0.945:
    print("🎖️  EXCELLENT: Very close to 95% target!")
else:
    print("✅ GOOD: Solid performance!")

print(f"\n✅ Key Optimizations Applied:")
print(f"   • Mixed precision training (30% faster)")
print(f"   • ResNet50 architecture with ImageNet weights")
print(f"   • Optimized augmentation strategy")
print(f"   • Enhanced TTA with 7 augmentations")
print(f"   • Better layer unfreezing (135+ layers)")
print(f"   • Cosine annealing with warmup")
print(f"   • Optimized hyperparameters")
print(f"   • Total epochs: {EPOCHS_PHASE1 + EPOCHS_PHASE2}")

# Save model
best_model.save(MODEL_SAVE_PATH)
print(f"\n💾 Model saved: {MODEL_SAVE_PATH}")
print(f"📊 Results saved: {results_file}")
print(f"📈 Evaluation plots saved: {plots_file}")
print(f"📈 Training curves saved: {training_plots_file}")

print("\n" + "=" * 70)
print("✅ OPTIMIZED TRAINING COMPLETE!")
print("=" * 70)
print(f"🎯 Final TTA Accuracy: {tta_acc * 100:.2f}%")
print("=" * 70)

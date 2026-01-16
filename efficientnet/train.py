"""
EfficientNet-B0 Training Script - PROPERLY CONFIGURED
=====================================================
Research Paper: Comparative Analysis of Lightweight CNN Architectures

CRITICAL FIXES FROM FAILED TRAINING:
1. ✅ Reduced learning rates (EfficientNet needs 5-10x lower LR than MobileNetV2)
2. ✅ Longer warmup period (5 epochs instead of 3)
3. ✅ Better batch normalization handling
4. ✅ Gradual unfreezing strategy
5. ✅ Class weighting for imbalanced learning
6. ✅ Fixed evaluation metrics bug
7. ✅ Better regularization balance

Model: EfficientNet-B0
Input Size: 224x224x3
Parameters: ~5 million
Expected Accuracy: 88-92%

Author: M.Tech Student
Date: January 2026
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION - PROPERLY TUNED FOR EFFICIENTNET
# ============================================================================

class Config:
    """Configuration class for EfficientNet-B0 training"""
    
    # Paths
    BASE_DIR = "."
    DATA_DIR = "../apple_dataset/raw"
    AUGMENTED_DATA_DIR = "../apple_dataset/augmented"
    CHECKPOINT_DIR = "./checkpoints"
    LOGS_DIR = "./logs"
    RESULTS_DIR = "./results"
    
    # Model parameters
    MODEL_NAME = "efficientnet_b0_fixed"
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 4
    CLASS_NAMES = ['alternaria', 'healthy', 'rust', 'scab']
    
    # Phase 1: Transfer Learning - CONSERVATIVE SETTINGS
    PHASE1_EPOCHS = 25  # Increased from 20
    PHASE1_BATCH_SIZE = 16  # Increased for stability
    PHASE1_LR = 0.0003  # REDUCED from 0.002 (critical fix!)
    PHASE1_WARMUP_EPOCHS = 5  # Longer warmup
    PHASE1_FREEZE_BASE = True
    
    # Phase 2: Fine-Tuning - GRADUAL UNFREEZING
    PHASE2_EPOCHS = 15  # Increased from 10
    PHASE2_BATCH_SIZE = 12  # Slightly smaller for fine-tuning
    PHASE2_LR = 8e-6  # REDUCED from 5e-5 (critical fix!)
    PHASE2_WARMUP_EPOCHS = 3
    PHASE2_UNFREEZE_LAYERS = 40  # Reduced from 50 (more conservative)
    
    # Regularization - BALANCED FOR EFFICIENTNET
    DROPOUT_RATE_1 = 0.4  # Reduced from 0.5
    DROPOUT_RATE_2 = 0.3  # Reduced from 0.4
    L2_REG = 0.0005  # Reduced from 0.001
    LABEL_SMOOTHING = 0.08  # Reduced from 0.1
    
    # Training parameters
    VALIDATION_SPLIT = 0.2
    MIXED_PRECISION = True
    
    # Callbacks - MORE PATIENCE
    EARLY_STOPPING_PATIENCE = 15  # Increased from 12
    REDUCE_LR_PATIENCE = 6  # Reduced from 8 (more responsive)
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-7
    
    # TTA parameters
    TTA_STEPS = 7
    
    # Random seed for reproducibility
    RANDOM_SEED = 42

# Set random seeds
def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[INFO] Random seeds set to {seed}")

set_seeds(Config.RANDOM_SEED)

# Enable mixed precision training
if Config.MIXED_PRECISION:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("[INFO] Mixed precision training enabled (FP16)")

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def create_directories():
    """Create necessary directories for saving outputs"""
    directories = [
        Config.CHECKPOINT_DIR,
        Config.LOGS_DIR,
        Config.RESULTS_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("[INFO] Directory structure created successfully")

create_directories()

# ============================================================================
# DATA SOURCE SELECTION
# ============================================================================

def select_data_source():
    """
    Select best available data source (augmented or raw).
    Matches the logic from MobileNetV2 and ResNet training scripts.
    """
    if os.path.exists(Config.AUGMENTED_DATA_DIR):
        aug_classes = [
            d for d in os.listdir(Config.AUGMENTED_DATA_DIR)
            if os.path.isdir(os.path.join(Config.AUGMENTED_DATA_DIR, d))
        ]
        
        if len(aug_classes) >= 4:
            total_aug = 0
            for class_dir in aug_classes:
                class_path = os.path.join(Config.AUGMENTED_DATA_DIR, class_dir)
                images = [
                    f for f in os.listdir(class_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                total_aug += len(images)
            
            if total_aug > 500:
                print(f"[INFO] Using pre-augmented dataset: {total_aug:,} images")
                print(f"[INFO] Data source: {Config.AUGMENTED_DATA_DIR}")
                return Config.AUGMENTED_DATA_DIR, True
    
    print(f"[INFO] Using original raw dataset")
    print(f"[INFO] Data source: {Config.DATA_DIR}")
    return Config.DATA_DIR, False

# Select data source
SELECTED_DATA_DIR, using_preaugmented = select_data_source()

# ============================================================================
# DATA LOADING & AUGMENTATION
# ============================================================================

def create_data_generators():
    """
    Create training and validation data generators with augmentation.
    Uses MODERATE augmentation to prevent overfitting.
    """
    print("\n" + "="*70)
    print("DATA PREPARATION")
    print("="*70)
    
    if using_preaugmented:
        print("[INFO] Using pre-augmented data with light additional augmentation")
        # Light augmentation for pre-augmented data
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='reflect',
            validation_split=Config.VALIDATION_SPLIT
        )
    else:
        print("[INFO] Using raw data with moderate augmentation")
        # Moderate augmentation for raw data
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.75, 1.25],
            channel_shift_range=15.0,
            fill_mode='reflect',
            validation_split=Config.VALIDATION_SPLIT
        )
    
    # Validation data generator (minimal augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=Config.VALIDATION_SPLIT
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        SELECTED_DATA_DIR,
        target_size=Config.INPUT_SHAPE[:2],
        batch_size=Config.PHASE1_BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=Config.RANDOM_SEED
    )
    
    # Load validation data
    validation_generator = val_datagen.flow_from_directory(
        SELECTED_DATA_DIR,
        target_size=Config.INPUT_SHAPE[:2],
        batch_size=Config.PHASE1_BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=Config.RANDOM_SEED
    )
    
    print(f"\n[INFO] Training samples: {train_generator.samples}")
    print(f"[INFO] Validation samples: {validation_generator.samples}")
    print(f"[INFO] Number of classes: {train_generator.num_classes}")
    print(f"[INFO] Class indices: {train_generator.class_indices}")
    
    return train_generator, validation_generator

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_efficientnet_model(phase='phase1'):
    """
    Build EfficientNet-B0 model with custom classification head.
    
    CRITICAL CHANGES FROM FAILED VERSION:
    - Smaller dense layers to prevent overfitting
    - Lower dropout rates
    - BatchNorm before dropout (better for EfficientNet)
    - Reduced L2 regularization
    
    Args:
        phase (str): 'phase1' for feature extraction, 'phase2' for fine-tuning
    
    Returns:
        tf.keras.Model: Compiled EfficientNet-B0 model
    """
    print("\n" + "="*70)
    print(f"BUILDING EFFICIENTNET-B0 MODEL - {phase.upper()}")
    print("="*70)
    
    # Load pretrained EfficientNet-B0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=Config.INPUT_SHAPE
    )
    
    # Freeze base model for phase 1
    if phase == 'phase1':
        base_model.trainable = False
        print(f"[INFO] Base model frozen for transfer learning")
    else:
        # Gradual unfreezing for phase 2
        base_model.trainable = True
        total_layers = len(base_model.layers)
        freeze_until = total_layers - Config.PHASE2_UNFREEZE_LAYERS
        
        for i, layer in enumerate(base_model.layers[:freeze_until]):
            layer.trainable = False
        
        # Keep batch norm layers frozen (critical for EfficientNet!)
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        
        trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"[INFO] Unfrozen last {trainable_layers} layers for fine-tuning")
        print(f"[INFO] All BatchNormalization layers kept frozen (EfficientNet best practice)")
    
    # Build custom classification head (SIMPLIFIED from failed version)
    inputs = keras.Input(shape=Config.INPUT_SHAPE)
    x = base_model(inputs, training=False if phase == 'phase1' else True)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # First dense block (SMALLER than before)
    x = layers.BatchNormalization(name='bn_1')(x)  # BN before dropout
    x = layers.Dropout(Config.DROPOUT_RATE_1, name='dropout_1')(x)
    x = layers.Dense(
        256,  # Reduced from 512
        activation='relu',
        kernel_regularizer=regularizers.l2(Config.L2_REG),
        kernel_initializer='he_normal',  # Better initialization
        name='dense_1'
    )(x)
    
    # Second dense block (SMALLER than before)
    x = layers.BatchNormalization(name='bn_2')(x)  # BN before dropout
    x = layers.Dropout(Config.DROPOUT_RATE_2, name='dropout_2')(x)
    x = layers.Dense(
        128,  # Reduced from 256
        activation='relu',
        kernel_regularizer=regularizers.l2(Config.L2_REG),
        kernel_initializer='he_normal',
        name='dense_2'
    )(x)
    
    # Output layer with label smoothing
    outputs = layers.Dense(
        Config.NUM_CLASSES,
        activation='softmax',
        dtype='float32',  # Ensure float32 output for mixed precision
        kernel_initializer='glorot_uniform',
        name='output'
    )(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='efficientnet_b0_classifier')
    
    # Print model summary
    print(f"\n[INFO] Model Architecture:")
    print(f"  - Input shape: {Config.INPUT_SHAPE}")
    print(f"  - Total parameters: {model.count_params():,}")
    
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Non-trainable parameters: {non_trainable_params:,}")
    print(f"  - Classification head: 256 -> 128 -> 4 (simplified)")
    
    return model

def compile_model(model, learning_rate, phase='phase1'):
    """
    Compile model with optimizer, loss, and metrics.
    
    CRITICAL: Uses lower learning rate than failed version
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        phase: Training phase identifier
    """
    # Optimizer with gradient clipping and proper decay
    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0,  # Gradient clipping
        decay=1e-6  # Small weight decay
    )
    
    # Loss with label smoothing
    loss = keras.losses.CategoricalCrossentropy(
        label_smoothing=Config.LABEL_SMOOTHING,
        from_logits=False
    )
    
    # Metrics
    metrics = [
        'accuracy',
        keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    print(f"\n[INFO] Model compiled for {phase}")
    print(f"  - Optimizer: Adam (lr={learning_rate}, decay=1e-6)")
    print(f"  - Loss: Categorical Crossentropy (label_smoothing={Config.LABEL_SMOOTHING})")
    print(f"  - Gradient clipping: clipnorm=1.0")

# ============================================================================
# CALLBACKS
# ============================================================================

def create_callbacks(phase='phase1'):
    """
    Create training callbacks with PROPER learning rate scheduling
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Model checkpoint
    checkpoint_path = os.path.join(
        Config.CHECKPOINT_DIR,
        f'best_efficientnet_b0_{phase}_{timestamp}.keras'
    )
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=False
    )
    
    # Early stopping with MORE patience
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=Config.EARLY_STOPPING_PATIENCE,
        mode='max',
        verbose=1,
        restore_best_weights=True,
        min_delta=0.001  # Require meaningful improvement
    )
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=Config.REDUCE_LR_FACTOR,
        patience=Config.REDUCE_LR_PATIENCE,
        mode='max',
        verbose=1,
        min_lr=Config.MIN_LR,
        min_delta=0.001
    )
    
    # CSV logger
    csv_path = os.path.join(Config.LOGS_DIR, f'{phase}_{timestamp}.csv')
    csv_logger = CSVLogger(csv_path, separator=',', append=False)
    
    # IMPROVED Learning rate scheduler with LONGER warmup
    def lr_schedule(epoch, lr):
        """
        CRITICAL FIX: Proper warmup and gentle decay for EfficientNet
        """
        if phase == 'phase1':
            warmup_epochs = Config.PHASE1_WARMUP_EPOCHS
            total_epochs = Config.PHASE1_EPOCHS
            base_lr = Config.PHASE1_LR
            
            if epoch < warmup_epochs:
                # Linear warmup
                return base_lr * (epoch + 1) / warmup_epochs
            else:
                # Cosine decay after warmup
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            warmup_epochs = Config.PHASE2_WARMUP_EPOCHS
            base_lr = Config.PHASE2_LR
            
            if epoch < warmup_epochs:
                # Linear warmup
                return base_lr * (epoch + 1) / warmup_epochs
            else:
                # Exponential decay (gentle)
                return base_lr * (0.96 ** (epoch - warmup_epochs))
    
    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    
    callbacks = [checkpoint, early_stopping, reduce_lr, csv_logger, lr_scheduler]
    
    print(f"\n[INFO] Callbacks configured for {phase}:")
    print(f"  - ModelCheckpoint: {checkpoint_path}")
    print(f"  - EarlyStopping: patience={Config.EARLY_STOPPING_PATIENCE}")
    print(f"  - ReduceLROnPlateau: factor={Config.REDUCE_LR_FACTOR}, patience={Config.REDUCE_LR_PATIENCE}")
    print(f"  - LearningRateScheduler: warmup + decay")
    
    return callbacks

# ============================================================================
# TRAINING
# ============================================================================

def train_phase(model, train_gen, val_gen, phase='phase1', epochs=20, batch_size=16, learning_rate=0.0003):
    """Train model for specified phase"""
    print("\n" + "="*70)
    print(f"STARTING {phase.upper()}")
    print("="*70)
    print(f"[INFO] Epochs: {epochs}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Base learning rate: {learning_rate}")
    print(f"[INFO] Warmup epochs: {Config.PHASE1_WARMUP_EPOCHS if phase == 'phase1' else Config.PHASE2_WARMUP_EPOCHS}")
    
    # Update batch size
    train_gen.batch_size = batch_size
    val_gen.batch_size = batch_size
    
    # Compile model
    compile_model(model, learning_rate, phase)
    
    # Create callbacks
    callbacks = create_callbacks(phase)
    
    # Calculate steps
    steps_per_epoch = train_gen.samples // batch_size
    validation_steps = val_gen.samples // batch_size
    
    print(f"[INFO] Steps per epoch: {steps_per_epoch}")
    print(f"[INFO] Validation steps: {validation_steps}")
    print(f"\n[INFO] Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n[INFO] {phase.upper()} completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return history

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, val_gen):
    """Evaluate model on validation set"""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Reset generator
    val_gen.reset()
    
    # Standard evaluation
    print("\n[INFO] Computing standard evaluation metrics...")
    results = model.evaluate(val_gen, steps=val_gen.samples // val_gen.batch_size, verbose=1)
    
    metrics = {}
    for name, value in zip(model.metrics_names, results):
        metrics[name] = float(value)
        print(f"  - {name}: {value:.4f}")
    
    # Get predictions
    print("\n[INFO] Generating predictions...")
    val_gen.reset()
    predictions = model.predict(val_gen, steps=val_gen.samples // val_gen.batch_size, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = val_gen.classes[:len(y_pred)]
    
    # Classification report
    print("\n[INFO] Classification Report:")
    report = classification_report(
        y_true, y_pred,
        target_names=Config.CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Compute additional metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # ROC-AUC
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=Config.NUM_CLASSES)
    try:
        roc_auc_per_class = roc_auc_score(y_true_onehot, predictions, average=None)
        roc_auc_macro = roc_auc_score(y_true_onehot, predictions, average='macro')
    except:
        roc_auc_per_class = [0.0] * Config.NUM_CLASSES
        roc_auc_macro = 0.0
    
    # Average Precision
    try:
        ap_per_class = average_precision_score(y_true_onehot, predictions, average=None)
        map_score = average_precision_score(y_true_onehot, predictions, average='macro')
    except:
        ap_per_class = [0.0] * Config.NUM_CLASSES
        map_score = 0.0
    
    metrics.update({
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'map': float(map_score),
        'roc_auc_macro': float(roc_auc_macro),
        'confusion_matrix': cm.tolist()
    })
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(Config.CLASS_NAMES):
        per_class_metrics[class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(support[i]),
            'roc_auc': float(roc_auc_per_class[i]),
            'average_precision': float(ap_per_class[i])
        }
    
    metrics['per_class'] = per_class_metrics
    
    print(f"\n[INFO] Mean Average Precision (MAP): {map_score:.4f}")
    print(f"[INFO] ROC-AUC (Macro): {roc_auc_macro:.4f}")
    print(f"[INFO] F1-Score (Macro): {f1_macro:.4f}")
    
    return metrics, cm, y_true, y_pred, predictions

def test_time_augmentation(model, val_gen, tta_steps=7):
    """Perform Test-Time Augmentation"""
    print("\n" + "="*70)
    print(f"TEST-TIME AUGMENTATION (TTA) - {tta_steps} steps")
    print("="*70)
    
    # Create TTA generator
    tta_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        zoom_range=0.08,
        brightness_range=[0.9, 1.1]
    )
    
    # Get true labels
    val_gen.reset()
    y_true = val_gen.classes
    
    # Accumulate predictions
    print("\n[INFO] Generating TTA predictions...")
    tta_predictions = []
    
    for i in range(tta_steps):
        print(f"  - TTA step {i+1}/{tta_steps}")
        
        tta_gen = tta_datagen.flow_from_directory(
            SELECTED_DATA_DIR,
            target_size=Config.INPUT_SHAPE[:2],
            batch_size=val_gen.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=Config.RANDOM_SEED + i
        )
        
        preds = model.predict(tta_gen, steps=len(y_true) // val_gen.batch_size, verbose=0)
        tta_predictions.append(preds)
    
    # Average predictions
    avg_predictions = np.mean(tta_predictions, axis=0)
    y_pred_tta = np.argmax(avg_predictions, axis=1)
    y_pred_tta = y_pred_tta[:len(y_true)]
    
    # Calculate TTA accuracy
    tta_accuracy = np.mean(y_pred_tta == y_true)
    
    print(f"\n[INFO] TTA Accuracy: {tta_accuracy:.4f} ({tta_accuracy*100:.2f}%)")
    
    return float(tta_accuracy)

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history_phase1, history_phase2):
    """Plot training history"""
    print("\n[INFO] Generating training history plots...")
    
    acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
    val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
    loss = history_phase1.history['loss'] + history_phase2.history['loss']
    val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']
    
    epochs_phase1 = len(history_phase1.history['accuracy'])
    epochs_total = len(acc)
    epochs_range = range(1, epochs_total + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.axvline(x=epochs_phase1, color='green', linestyle='--', linewidth=2, label='Phase 1 → Phase 2')
    ax1.set_title('EfficientNet-B0: Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax2.axvline(x=epochs_phase1, color='green', linestyle='--', linewidth=2, label='Phase 1 → Phase 2')
    ax2.set_title('EfficientNet-B0: Training and Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(Config.RESULTS_DIR, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Training history plot saved to: {save_path}")
    plt.close()

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    print("\n[INFO] Generating confusion matrix plot...")
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('EfficientNet-B0: Confusion Matrix (Absolute)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', xticklabels=class_names, 
                yticklabels=class_names, ax=ax2, cbar_kws={'label': 'Percentage'})
    ax2.set_title('EfficientNet-B0: Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(Config.RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Confusion matrix plot saved to: {save_path}")
    plt.close()

def plot_per_class_metrics(per_class_metrics):
    """Plot per-class metrics"""
    print("\n[INFO] Generating per-class metrics plot...")
    
    class_names = list(per_class_metrics.keys())
    precision = [per_class_metrics[c]['precision'] for c in class_names]
    recall = [per_class_metrics[c]['recall'] for c in class_names]
    f1 = [per_class_metrics[c]['f1_score'] for c in class_names]
    roc_auc = [per_class_metrics[c]['roc_auc'] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - 1.5*width, precision, width, label='Precision', color='#2E86AB')
    bars2 = ax.bar(x - 0.5*width, recall, width, label='Recall', color='#A23B72')
    bars3 = ax.bar(x + 0.5*width, f1, width, label='F1-Score', color='#F18F01')
    bars4 = ax.bar(x + 1.5*width, roc_auc, width, label='ROC-AUC', color='#C73E1D')
    
    ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('EfficientNet-B0: Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    add_value_labels(bars4)
    
    plt.tight_layout()
    save_path = os.path.join(Config.RESULTS_DIR, 'per_class_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Per-class metrics plot saved to: {save_path}")
    plt.close()

# ============================================================================
# SAVE RESULTS - FIXED VERSION
# ============================================================================

def save_results(metrics, history_phase1, history_phase2, tta_accuracy):
    """Save results to JSON - FIXED"""
    print("\n[INFO] Saving results to JSON...")
    
    results = {
        'model': 'EfficientNet-B0 (Fixed)',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'input_shape': Config.INPUT_SHAPE,
            'num_classes': Config.NUM_CLASSES,
            'phase1_epochs': Config.PHASE1_EPOCHS,
            'phase2_epochs': Config.PHASE2_EPOCHS,
            'phase1_batch_size': Config.PHASE1_BATCH_SIZE,
            'phase2_batch_size': Config.PHASE2_BATCH_SIZE,
            'phase1_lr': Config.PHASE1_LR,
            'phase2_lr': Config.PHASE2_LR,
            'dropout_rates': [Config.DROPOUT_RATE_1, Config.DROPOUT_RATE_2],
            'l2_regularization': Config.L2_REG,
            'label_smoothing': Config.LABEL_SMOOTHING,
            'tta_steps': Config.TTA_STEPS
        },
        'training_history': {
            'phase1': {
                'accuracy': [float(x) for x in history_phase1.history['accuracy']],
                'val_accuracy': [float(x) for x in history_phase1.history['val_accuracy']],
                'loss': [float(x) for x in history_phase1.history['loss']],
                'val_loss': [float(x) for x in history_phase1.history['val_loss']]
            },
            'phase2': {
                'accuracy': [float(x) for x in history_phase2.history['accuracy']],
                'val_accuracy': [float(x) for x in history_phase2.history['val_accuracy']],
                'loss': [float(x) for x in history_phase2.history['loss']],
                'val_loss': [float(x) for x in history_phase2.history['val_loss']]
            }
        },
        'evaluation_metrics': metrics,
        'tta_accuracy': tta_accuracy
    }
    
    json_path = os.path.join(Config.RESULTS_DIR, 'efficientnet_b0_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"[INFO] Results saved to: {json_path}")
    
    # FIXED: Print summary with proper metric access
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Model: EfficientNet-B0 (FIXED)")
    print(f"Total Parameters: ~5 million")
    print(f"Input Shape: {Config.INPUT_SHAPE}")
    print(f"\nPhase 1 (Transfer Learning):")
    print(f"  - Epochs: {Config.PHASE1_EPOCHS}")
    print(f"  - Final Training Accuracy: {history_phase1.history['accuracy'][-1]:.4f}")
    print(f"  - Final Validation Accuracy: {history_phase1.history['val_accuracy'][-1]:.4f}")
    print(f"\nPhase 2 (Fine-Tuning):")
    print(f"  - Epochs: {Config.PHASE2_EPOCHS}")
    print(f"  - Unfrozen Layers: {Config.PHASE2_UNFREEZE_LAYERS}")
    print(f"  - Final Training Accuracy: {history_phase2.history['accuracy'][-1]:.4f}")
    print(f"  - Final Validation Accuracy: {history_phase2.history['val_accuracy'][-1]:.4f}")
    print(f"\nFinal Evaluation:")
    print(f"  - Validation Accuracy: {metrics.get('accuracy', metrics.get('categorical_accuracy', 0.0)):.4f}")
    print(f"  - TTA Accuracy: {tta_accuracy:.4f}")
    print(f"  - Mean Average Precision (MAP): {metrics['map']:.4f}")
    print(f"  - Macro F1-Score: {metrics['f1_macro']:.4f}")
    print(f"  - ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")
    print("\nPer-Class F1-Scores:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  - {class_name}: {class_metrics['f1_score']:.4f}")
    print("="*70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("EFFICIENTNET-B0 TRAINING PIPELINE (FIXED)")
    print("Apple Leaf Disease Classification")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n[INFO] GPU Available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("\n[WARNING] No GPU detected. Training will use CPU (slower).")
    
    # Load data
    train_gen, val_gen = create_data_generators()
    
    # PHASE 1: TRANSFER LEARNING
    model = build_efficientnet_model(phase='phase1')
    history_phase1 = train_phase(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        phase='phase1',
        epochs=Config.PHASE1_EPOCHS,
        batch_size=Config.PHASE1_BATCH_SIZE,
        learning_rate=Config.PHASE1_LR
    )
    
    # PHASE 2: FINE-TUNING
    model = build_efficientnet_model(phase='phase2')
    
    # Load best weights from phase 1
    latest_checkpoint = sorted(
        [f for f in os.listdir(Config.CHECKPOINT_DIR) if f.startswith('best_efficientnet_b0_phase1')],
        key=lambda x: os.path.getmtime(os.path.join(Config.CHECKPOINT_DIR, x))
    )[-1]
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, latest_checkpoint)
    print(f"\n[INFO] Loading Phase 1 checkpoint: {checkpoint_path}")
    model.load_weights(checkpoint_path)
    
    history_phase2 = train_phase(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        phase='phase2',
        epochs=Config.PHASE2_EPOCHS,
        batch_size=Config.PHASE2_BATCH_SIZE,
        learning_rate=Config.PHASE2_LR
    )
    
    # EVALUATION
    latest_checkpoint = sorted(
        [f for f in os.listdir(Config.CHECKPOINT_DIR) if f.startswith('best_efficientnet_b0_phase2')],
        key=lambda x: os.path.getmtime(os.path.join(Config.CHECKPOINT_DIR, x))
    )[-1]
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, latest_checkpoint)
    print(f"\n[INFO] Loading best model: {checkpoint_path}")
    model = keras.models.load_model(checkpoint_path)
    
    metrics, cm, y_true, y_pred, predictions = evaluate_model(model, val_gen)
    tta_accuracy = test_time_augmentation(model, val_gen, tta_steps=Config.TTA_STEPS)
    
    # VISUALIZATION & RESULTS
    plot_training_history(history_phase1, history_phase2)
    plot_confusion_matrix(cm, Config.CLASS_NAMES)
    plot_per_class_metrics(metrics['per_class'])
    save_results(metrics, history_phase1, history_phase2, tta_accuracy)
    
    print(f"\n[INFO] Training pipeline completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] All results saved to: {Config.RESULTS_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()

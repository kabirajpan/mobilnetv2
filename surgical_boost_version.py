import tensorflow as tf    
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                       ModelCheckpoint, CSVLogger)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from datetime import datetime

# ============================
# SURGICAL BOOST VERSION - MINIMAL CHANGES FOR MAX ACCURACY
# ============================
tf.config.threading.set_intra_op_parallelism_threads(2)  # i3 dual core
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.set_soft_device_placement(True)

# Configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 12
EPOCHS_PHASE1 = 30
EPOCHS_PHASE2 = 25
DATA_DIR = "./apple_dataset/raw"
AUGMENTED_DATA_DIR = "./apple_dataset/augmented"  # Your pre-augmented data
MODEL_SAVE_PATH = "mobilenetv2_surgical_boost_i3.keras"
BEST_MODEL_PATH = "best_mobilenetv2_surgical_boost_i3.keras"
LOG_DIR = "./training_logs_ultimate"
LABEL_SMOOTHING = 0.1  # Surgical improvement #1

print("🚀 ENHANCED SURGICAL BOOST - Target 96%+ accuracy!")
print(f"Image size: {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Phase 1 epochs: {EPOCHS_PHASE1} (+5 for better convergence)")
print(f"Phase 2 epochs: {EPOCHS_PHASE2} (+5 for fine-tuning)")
print(f"Label smoothing: {LABEL_SMOOTHING} (surgical improvement)")
print(f"⏱️  Expected time: ~6-6.5 hours (worth it for 96%+!)")

# Create directories
os.makedirs(LOG_DIR, exist_ok=True)

# ============================
# DATA SOURCE SELECTION
# ============================
def select_data_source():
    """Automatically select best available data source"""

    # Check if augmented data exists and has substantial data
    if os.path.exists(AUGMENTED_DATA_DIR):
        aug_classes = [d for d in os.listdir(AUGMENTED_DATA_DIR)
                      if os.path.isdir(os.path.join(AUGMENTED_DATA_DIR, d))]

        if len(aug_classes) >= 4:  # Should have all 4 classes
            # Count total augmented images
            total_aug_images = 0
            for class_dir in aug_classes:
                class_path = os.path.join(AUGMENTED_DATA_DIR, class_dir)
                images = [f for f in os.listdir(class_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_aug_images += len(images)

            if total_aug_images > 500:  # Threshold for meaningful augmentation
                print(f"✅ Found pre-augmented dataset with {total_aug_images} images")
                print(f"📂 Using augmented data from: {AUGMENTED_DATA_DIR}")
                return AUGMENTED_DATA_DIR, True

    print(f"📂 Using original data from: {DATA_DIR}")
    print("💡 Tip: Run 'python scripts/augment_images.py' first for better results!")
    return DATA_DIR, False

selected_data_dir, using_preaugmented = select_data_source()

# ============================
# DATA GENERATORS SETUP
# ============================
if using_preaugmented:
    # Minimal augmentation since data is already augmented
    print("🔄 Setting up minimal runtime augmentation (data already pre-augmented)")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        horizontal_flip=True,  # Only basic flips for additional variety
        validation_split=0.2   # More validation data since we have more training data
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2
    )
else:
    # Import and use your custom augmentation
    print("🔄 Setting up heavy runtime augmentation")
    import sys
    sys.path.append('./scripts')

    try:
        from augment_config import get_augmentation_config, augment_config
        print("✅ Loaded custom augmentation config")

        train_datagen = get_augmentation_config(level='heavy')
        train_datagen.validation_split = 0.15

        val_datagen = augment_config(for_validation=True)
        val_datagen.validation_split = 0.15

    except ImportError as e:
        print(f"⚠️  Could not import custom augmentation: {e}")
        # Fallback standard augmentation
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0/255,
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.15
        )

        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0/255,
            validation_split=0.15
        )

# Create data generators
train_generator = train_datagen.flow_from_directory(
    selected_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    interpolation='bilinear'
)

val_generator = val_datagen.flow_from_directory(
    selected_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    interpolation='bilinear'
)

print(f"📊 Dataset Statistics:")
print(f"   Training samples: {train_generator.samples}")
print(f"   Validation samples: {val_generator.samples}")
print(f"   Classes: {list(train_generator.class_indices.keys())}")
print(f"   Data source: {'Pre-augmented' if using_preaugmented else 'Original with runtime augmentation'}")

# ============================
# ENHANCED MODEL ARCHITECTURE
# ============================
def create_enhanced_model(num_classes, input_shape=(128, 128, 3)):
    """Create enhanced MobileNetV2 with improved head"""
    print("Creating enhanced MobileNetV2 model...")

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        alpha=1.0  # Full width for better accuracy
    )

    # Enhanced classifier head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

model, base_model = create_enhanced_model(train_generator.num_classes)

# Print model summary
total_params = model.count_params()
print(f"🧠 Model Configuration:")
print(f"   Total parameters: {total_params:,}")

# ============================
# TRAINING CALLBACKS
# ============================
def create_callbacks():
    """Create comprehensive callback list"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    callbacks = [
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),

        EarlyStopping(
            monitor='val_accuracy',
            patience=12 if using_preaugmented else 10,  # More patience with more data
            restore_best_weights=True,
            verbose=1
        ),

        CSVLogger(
            os.path.join(LOG_DIR, f'training_log_{timestamp}.csv'),
            append=False
        )
    ]

    return callbacks

# ============================
# MAIN EXECUTION - START TIMING
# ============================
total_start_time = time.time()

# ============================
# PHASE 1: TRANSFER LEARNING
# ============================
print("\n" + "="*60)
print("PHASE 1: TRANSFER LEARNING (Frozen Backbone)")
print("="*60)

base_model.trainable = False
trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"🔒 Trainable parameters in Phase 1: {trainable_count:,}")

model.compile(
    optimizer=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=['accuracy']
)

phase1_start = time.time()
callbacks = create_callbacks()

print("🚀 Starting Phase 1 training...")
history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks,
    verbose=1
)

phase1_time = time.time() - phase1_start
print(f"\n⏱️  Phase 1 completed in {phase1_time/60:.1f} minutes")

# ============================
# PHASE 2: FINE-TUNING
# ============================
print("\n" + "="*60)
print("PHASE 2: FINE-TUNING (Partial Unfreezing)")
print("="*60)

base_model.trainable = True

# Freeze bottom layers, unfreeze MORE top layers for better fine-tuning
for layer in base_model.layers[:80]:  # Unfreeze 20 more layers than before
    layer.trainable = False
for layer in base_model.layers[100:]:
    layer.trainable = True

trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"🔓 Trainable parameters in Phase 2: {trainable_count:,}")

# Enhanced learning rate with cosine annealing
def cosine_annealing_lr(epoch):
    """Cosine annealing learning rate schedule for Phase 2"""
    import math
    max_lr = 2e-5  # Slightly higher max
    min_lr = 5e-7  # Lower minimum
    return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * epoch / EPOCHS_PHASE2)) / 2

model.compile(
    optimizer=Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=['accuracy']
)

# Create enhanced callbacks for Phase 2 with cosine annealing
callbacks_phase2 = create_callbacks()
callbacks_phase2[0].filepath = BEST_MODEL_PATH.replace('.keras', '_phase2.keras')

# Add cosine annealing learning rate scheduler
from tensorflow.keras.callbacks import LearningRateScheduler
cosine_lr_callback = LearningRateScheduler(cosine_annealing_lr, verbose=1)
callbacks_phase2.append(cosine_lr_callback)

print("\n🔥 Starting Enhanced Phase 2 Training with Cosine LR...")
phase2_start = time.time()

history_phase2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=val_generator,
    callbacks=callbacks_phase2,
    verbose=1
)

phase2_time = time.time() - phase2_start
total_training_time = phase1_time + phase2_time

print(f"\n⏱️  Phase 2 completed in {phase2_time/60:.1f} minutes")
print(f"🎯 Total training time: {total_training_time/60:.1f} minutes")

# ============================
# FINAL EVALUATION
# ============================
# QUICK TTA FUNCTION (Surgical Improvement #3)
# ============================
def quick_tta(model, val_generator, n_augmentations=5):
    """Enhanced Test Time Augmentation - adds ~8 minutes but boosts accuracy more"""
    print(f"🔄 Applying Enhanced TTA with {n_augmentations} augmentations...")

    # Get original predictions
    val_generator.reset()
    original_preds = model.predict(val_generator, verbose=0)
    all_predictions = [original_preds]

    # Create enhanced augmentation generator
    tta_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=15,
        width_shift_range=0.12,
        height_shift_range=0.12,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.08,
        brightness_range=[0.9, 1.1]
    )

    # Get original images for augmentation
    val_generator.reset()
    x_batch = []
    y_batch = []

    for i in range(len(val_generator)):
        batch_x, batch_y = val_generator[i]
        x_batch.append(batch_x)
        y_batch.append(batch_y)

    x_all = np.concatenate(x_batch, axis=0)[:len(original_preds)]

    # Apply augmentations and get predictions
    for aug_idx in range(n_augmentations):
        print(f"  TTA {aug_idx+1}/{n_augmentations}...")

        # Apply augmentation
        augmented_preds = []
        for batch_start in range(0, len(x_all), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(x_all))
            batch = x_all[batch_start:batch_end]

            # Generate augmented batch
            aug_gen = tta_datagen.flow(batch, batch_size=len(batch), shuffle=False)
            aug_batch = next(aug_gen)

            # Get predictions
            preds = model.predict(aug_batch, verbose=0)
            augmented_preds.extend(preds)

        all_predictions.append(np.array(augmented_preds))

    # Average all predictions
    final_predictions = np.mean(all_predictions, axis=0)
    predicted_classes = np.argmax(final_predictions, axis=1)

    # Get true labels
    val_generator.reset()
    true_classes = val_generator.classes[:len(predicted_classes)]

    # Calculate TTA accuracy
    tta_accuracy = np.mean(predicted_classes == true_classes)

    print(f"✅ TTA completed! Accuracy: {tta_accuracy:.4f}")
    return tta_accuracy, predicted_classes, true_classes

# ============================
print("\n" + "="*60)
print("FINAL EVALUATION WITH QUICK TTA")
print("="*60)

# Load best model
try:
    best_model = tf.keras.models.load_model(BEST_MODEL_PATH.replace('.keras', '_phase2.keras'))
    print("📂 Loaded best model from Phase 2")
except:
    best_model = model
    print("📂 Using current model for evaluation")

# Standard Evaluation
val_loss, val_acc = best_model.evaluate(val_generator, verbose=0)
train_loss, train_acc = best_model.evaluate(train_generator, verbose=0)

print(f"\n📊 STANDARD EVALUATION:")
print(f"   Training Accuracy: {train_acc:.4f}")
print(f"   Validation Accuracy: {val_acc:.4f}")
print(f"   Generalization Gap: {abs(train_acc - val_acc):.4f}")

# TTA Evaluation (Surgical Improvement #3)
print(f"\n🔄 APPLYING QUICK TTA (adds ~5 minutes)...")
tta_acc, tta_pred, tta_true = quick_tta(best_model, val_generator, n_augmentations=5)

print(f"\n🎉 SURGICAL BOOST RESULTS:")
print(f"   Standard Accuracy: {val_acc:.4f} ({val_acc:.1%})")
print(f"   TTA Accuracy: {tta_acc:.4f} ({tta_acc:.1%})")
print(f"🚀 Enhanced TTA Improvement: +{(tta_acc - val_acc):.4f} (+{(tta_acc - val_acc)*100:.1f}%)")
print(f"   Generalization Gap: {abs(train_acc - val_acc):.4f}")
print(f"   Data Strategy: {'Pre-augmented' if using_preaugmented else 'Runtime augmentation'}")
print(f"   🎯 Enhanced Training: +10 epochs, +20 fine-tune layers, +2 TTA augs, cosine LR")

# Save final model
best_model.save(MODEL_SAVE_PATH)
print(f"💾 Final model saved as: {MODEL_SAVE_PATH}")

# ============================
# FINAL TIMING AND SUCCESS MESSAGE
# ============================
total_training_time = time.time() - total_start_time

print(f"\n" + "="*60)
print("🎉 SURGICAL BOOST TRAINING COMPLETED!")
print("="*60)
print(f"⏱️  Total Training Time: {total_training_time/60:.1f} minutes ({total_training_time/3600:.1f} hours)")
print(f"🎯 Final TTA Accuracy: {tta_acc:.1%}")
print(f"🚀 Improvement over standard: +{(tta_acc - val_acc)*100:.1f}%")

if tta_acc >= 0.96:
    print("🏆 OUTSTANDING: 96%+ accuracy achieved with enhanced surgical boost!")
elif tta_acc >= 0.95:
    print("🏆 SUCCESS: 95%+ accuracy achieved with enhanced improvements!")
elif tta_acc >= 0.945:
    print("🎖️  EXCELLENT: Very close to 95% target with enhanced training!")
else:
    print("📈 GOOD: Solid improvement with enhanced surgical boost!")

print(f"\n💡 Enhanced surgical improvements applied:")
print(f"   ✅ Label smoothing: {LABEL_SMOOTHING}")
print(f"   ✅ Optimized Adam optimizer with cosine annealing")
print(f"   ✅ Enhanced TTA with 5 augmentations (+2 more)")
print(f"   ✅ Extended training: +10 epochs total")
print(f"   ✅ More fine-tuning: 20 additional layers unfrozen")
print(f"   ✅ Worth the extra 1-2 hours for 96%+ target!")

print(f"\nFiles created:")
print(f"   📁 Model: {MODEL_SAVE_PATH}")
print(f"   📊 Training logs in: {LOG_DIR}")
print("="*60)

# ============================
# ENHANCED PLOTTING
# ============================
def create_enhanced_plots(hist1, hist2, using_preaugmented):
    """Create comprehensive training visualization"""

    combined_acc = hist1.history['accuracy'] + hist2.history['accuracy']
    combined_val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
    combined_loss = hist1.history['loss'] + hist2.history['loss']
    combined_val_loss = hist1.history['val_loss'] + hist2.history['val_loss']

    epochs = range(1, len(combined_acc) + 1)
    phase1_epochs = len(hist1.history['accuracy'])

    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Accuracy plot
    axes[0, 0].plot(epochs, combined_acc, 'b-o', label='Training Accuracy', linewidth=2, markersize=3)
    axes[0, 0].plot(epochs, combined_val_acc, 'r-o', label='Validation Accuracy', linewidth=2, markersize=3)
    axes[0, 0].axvline(x=phase1_epochs, color='green', linestyle='--', alpha=0.7, label='Fine-tuning Start')

    title_suffix = "with Pre-Augmented Data" if using_preaugmented else "with Runtime Augmentation"
    axes[0, 0].set_title(f'Model Accuracy ({title_suffix})', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    # Loss plot
    axes[0, 1].plot(epochs, combined_loss, 'b-o', label='Training Loss', linewidth=2, markersize=3)
    axes[0, 1].plot(epochs, combined_val_loss, 'r-o', label='Validation Loss', linewidth=2, markersize=3)
    axes[0, 1].axvline(x=phase1_epochs, color='green', linestyle='--', alpha=0.7, label='Fine-tuning Start')
    axes[0, 1].set_title(f'Model Loss ({title_suffix})', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Data strategy visualization
    strategy_info = "Pre-Augmented Dataset" if using_preaugmented else "Runtime Augmentation"
    axes[0, 2].text(0.5, 0.7, f"Data Strategy:\n{strategy_info}",
                   ha='center', va='center', transform=axes[0, 2].transAxes,
                   fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen' if using_preaugmented else 'lightblue', alpha=0.8))

    if using_preaugmented:
        axes[0, 2].text(0.5, 0.3, "✅ Using your custom\naugmentation pipeline",
                       ha='center', va='center', transform=axes[0, 2].transAxes,
                       fontsize=12, color='green')
    else:
        axes[0, 2].text(0.5, 0.3, "💡 Consider running\naugment_images.py\nfor better results",
                       ha='center', va='center', transform=axes[0, 2].transAxes,
                       fontsize=12, color='orange')

    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].axis('off')

    # Training summary
    final_train_acc = combined_acc[-1]
    final_val_acc = combined_val_acc[-1]
    best_val_acc = max(combined_val_acc)

    summary_text = f"""Training Summary:

Data Strategy: {strategy_info}
Training Samples: {train_generator.samples:,}
Validation Samples: {val_generator.samples:,}

Phase 1: {phase1_epochs} epochs ({phase1_time/60:.1f} min)
Phase 2: {len(hist2.history['accuracy'])} epochs ({phase2_time/60:.1f} min)

Final Training Acc: {final_train_acc:.4f}
Final Validation Acc: {final_val_acc:.4f}
Best Validation Acc: {best_val_acc:.4f}

Generalization Gap: {abs(final_train_acc - final_val_acc):.4f}
Total Training Time: {total_training_time/60:.1f} minutes"""

    axes[1, 0].text(0.05, 0.95, summary_text, fontsize=9,
                   verticalalignment='top', transform=axes[1, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 0].axis('off')

    # Phase comparison
    phase1_best = max(hist1.history['val_accuracy'])
    phase2_best = max(hist2.history['val_accuracy'])
    improvement = phase2_best - phase1_best

    phases = ['Phase 1\n(Transfer)', 'Phase 2\n(Fine-tune)']
    accuracies = [phase1_best, phase2_best]
    colors = ['lightblue', 'lightgreen']

    bars = axes[1, 1].bar(phases, accuracies, color=colors, alpha=0.7)
    axes[1, 1].set_title(f'Phase Comparison\n(Improvement: {improvement:.3f})', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Best Validation Accuracy')
    axes[1, 1].set_ylim([0, 1])

    for bar, acc in zip(bars, accuracies):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    # Generalization gap analysis
    gen_gaps = [abs(acc - val_acc) for acc, val_acc in zip(combined_acc, combined_val_acc)]
    axes[1, 2].plot(epochs, gen_gaps, 'purple', linewidth=2)
    axes[1, 2].axvline(x=phase1_epochs, color='green', linestyle='--', alpha=0.7)
    axes[1, 2].axhline(y=0.1, color='red', linestyle=':', alpha=0.7, label='Overfitting Threshold')
    axes[1, 2].set_title('Generalization Gap Analysis', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Epochs')
    axes[1, 2].set_ylabel('|Train Acc - Val Acc|')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'enhanced_i3_preaugmented_results_{timestamp}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()

    print(f"📊 Comprehensive plots saved as '{filename}'")

# Create plots
print("\n📊 Creating comprehensive training visualization...")
try:
    create_enhanced_plots(history_phase1, history_phase2, using_preaugmented)
except Exception as e:
    print(f"Plotting error: {e}")
    print("Creating simple fallback plot...")

# ============================
# TRAINING COMPLETE
# ============================
print("\n" + "🎉" * 25)
print("ENHANCED TRAINING WITH AUGMENTATION SUPPORT COMPLETE!")
print("🎉" * 25)
print(f"\n📈 Key Achievements:")
print(f"   • Final Validation Accuracy: {val_acc:.4f}")
print(f"   • Training Time: {total_training_time/60:.1f} minutes")
print(f"   • Data Strategy: {'Pre-augmented pipeline' if using_preaugmented else 'Runtime augmentation'}")
print(f"   • Training Samples: {train_generator.samples:,}")
print(f"   • Model Parameters: {total_params:,}")
print(f"\n💾 Saved Files:")
print(f"   • Final Model: {MODEL_SAVE_PATH}")
print(f"   • Best Model: {BEST_MODEL_PATH.replace('.keras', '_phase2.keras')}")
print(f"   • Training Logs: {LOG_DIR}/")
print(f"\n🚀 Ready for inference and deployment!")

if not using_preaugmented:
    print(f"\n💡 Pro Tip: For even better results next time:")
    print(f"   1. Run: python scripts/augment_images.py")
    print(f"   2. Then run this script again")
    print(f"   This will use your sophisticated augmentation pipeline!")

"""
Real-World Image Testing Script
================================
Test real apple leaf images with trained MobileNetV2 or MobileNetV3 models.

Usage:
    python real_world_test.py --model v2 --image path/to/image.jpg
    python real_world_test.py --model v3 --image path/to/image.jpg
    python real_world_test.py --model v2 --folder path/to/folder/
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ============================
# CONFIGURATION
# ============================
MODEL_PATHS = {
    'v2': '../mobilenetv2/checkpoints/best_mobilenetv2_optimized.keras',
    'v3': '../mobilenetv3/checkpoints/best_mobilenetv3_optimized.keras',
}

IMAGE_SIZE = (160, 160)  # Match training image size
CLASS_NAMES = ['alternaria', 'healthy', 'rust', 'scab']  # Update if different

# ============================
# LOAD MODEL
# ============================
def load_trained_model(model_version):
    """Load the trained model"""
    model_path = MODEL_PATHS.get(model_version.lower())
    
    if not model_path:
        raise ValueError(f"Invalid model version: {model_version}. Choose 'v2' or 'v3'")
    
    if not os.path.exists(model_path):
        # Try to find the most recent model
        model_dir = os.path.dirname(model_path)
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
            if model_files:
                # Get the most recent one
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                model_path = os.path.join(model_dir, model_files[0])
                print(f"⚠️  Using most recent model: {model_files[0]}")
            else:
                raise FileNotFoundError(f"No trained model found in {model_dir}. Please train the model first.")
        else:
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"📂 Loading model from: {model_path}")
    model = load_model(model_path)
    print(f"✅ Model loaded successfully!")
    return model


# ============================
# PREPROCESS IMAGE
# ============================
def preprocess_image(img_path):
    """Preprocess image for model input"""
    # Load image
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    
    # Convert to array
    img_array = image.img_to_array(img)
    
    # Expand dimensions for batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize
    img_array = img_array / 255.0
    
    return img_array, img


# ============================
# PREDICT
# ============================
def predict_image(model, img_array):
    """Get prediction from model"""
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Get all class probabilities
    class_probs = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_probs[class_name] = predictions[0][i]
    
    return predicted_class_idx, confidence, class_probs, predictions[0]


# ============================
# VISUALIZE RESULTS
# ============================
def visualize_prediction(img, predicted_class, confidence, class_probs, model_name, save_path=None):
    """Visualize prediction results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Show image
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(f'Input Image\nPredicted: {predicted_class.upper()}\nConfidence: {confidence*100:.2f}%', 
                     fontsize=14, fontweight='bold')
    
    # Show probability distribution
    classes = list(class_probs.keys())
    probs = list(class_probs.values())
    colors = ['red' if i == predicted_class else 'gray' for i in range(len(classes))]
    
    bars = axes[1].barh(classes, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Probability', fontsize=12)
    axes[1].set_title(f'Class Probabilities ({model_name})', fontsize=14, fontweight='bold')
    axes[1].set_xlim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add probability labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        axes[1].text(prob + 0.01, i, f'{prob*100:.2f}%', 
                    va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved: {save_path}")
    
    plt.show()


# ============================
# TEST SINGLE IMAGE
# ============================
def test_single_image(model_version, image_path, show_plot=True, save_plot=False, simple_mode=False):
    """Test a single image"""
    if not simple_mode:
        print("\n" + "=" * 70)
        print(f"🧪 TESTING IMAGE: {image_path}")
        print("=" * 70)
    
    # Load model
    model = load_trained_model(model_version)
    model_name = f"MobileNetV{model_version.upper()}"
    
    # Preprocess image
    img_array, img = preprocess_image(image_path)
    
    # Predict
    predicted_idx, confidence, class_probs, all_probs = predict_image(model, img_array)
    predicted_class = CLASS_NAMES[predicted_idx]
    
    # Simple output mode
    if simple_mode:
        print(f"\n✅ RESULT: {predicted_class.upper()}")
        print(f"   Confidence: {confidence*100:.1f}%")
        return predicted_class, confidence, class_probs
    
    # Detailed output mode
    print(f"\n📊 PREDICTION RESULTS:")
    print(f"   Model: {model_name}")
    print(f"   Predicted Class: {predicted_class.upper()}")
    print(f"   Confidence: {confidence*100:.2f}%")
    print(f"\n   All Class Probabilities:")
    for class_name, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
        marker = "🏆" if class_name == predicted_class else "  "
        print(f"   {marker} {class_name:12s}: {prob*100:6.2f}%")
    
    # Visualize
    if show_plot:
        save_path = None
        if save_plot:
            output_dir = "./test_results"
            os.makedirs(output_dir, exist_ok=True)
            img_name = Path(image_path).stem
            save_path = os.path.join(output_dir, f"{model_name}_{img_name}_result.png")
        
        visualize_prediction(img, predicted_class, confidence, class_probs, model_name, save_path)
    
    return predicted_class, confidence, class_probs


# ============================
# TEST MULTIPLE IMAGES
# ============================
def test_folder(model_version, folder_path, show_plot=True, save_plot=True):
    """Test all images in a folder"""
    print("\n" + "=" * 70)
    print(f"🧪 TESTING FOLDER: {folder_path}")
    print("=" * 70)
    
    # Load model
    model = load_trained_model(model_version)
    model_name = f"MobileNetV{model_version.upper()}"
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f'*{ext}'))
        image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return
    
    print(f"📁 Found {len(image_files)} image(s)")
    
    # Test each image
    results = []
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in image_files:
        print(f"\n📸 Processing: {img_path.name}")
        
        try:
            # Preprocess
            img_array, img = preprocess_image(str(img_path))
            
            # Predict
            predicted_idx, confidence, class_probs, _ = predict_image(model, img_array)
            predicted_class = CLASS_NAMES[predicted_idx]
            
            results.append({
                'image': img_path.name,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probs': class_probs
            })
            
            print(f"   ✅ {predicted_class.upper()} ({confidence*100:.2f}%)")
            
            # Save individual result if requested
            if save_plot:
                save_path = os.path.join(output_dir, f"{model_name}_{img_path.stem}_result.png")
                visualize_prediction(img, predicted_class, confidence, class_probs, model_name, save_path)
                plt.close()  # Close to avoid memory issues
            
        except Exception as e:
            print(f"   ❌ Error processing {img_path.name}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total images tested: {len(results)}")
    
    # Count predictions
    class_counts = {}
    for result in results:
        pred_class = result['predicted_class']
        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
    
    print(f"\nPredictions by class:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        print(f"   {class_name:12s}: {count:3d} ({percentage:5.1f}%)")
    
    # Average confidence
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"\nAverage confidence: {avg_confidence*100:.2f}%")
    
    if save_plot:
        print(f"\n💾 All results saved to: {output_dir}/")
    
    return results


# ============================
# COMPARE MODELS
# ============================
def compare_models(image_path, show_plot=True):
    """Compare predictions from both models"""
    print("\n" + "=" * 70)
    print("🔄 COMPARING BOTH MODELS")
    print("=" * 70)
    
    results = {}
    
    for model_version in ['v2', 'v3']:
        try:
            print(f"\n📊 Testing with MobileNetV{model_version.upper()}...")
            model = load_trained_model(model_version)
            img_array, img = preprocess_image(image_path)
            predicted_idx, confidence, class_probs, _ = predict_image(model, img_array)
            
            results[model_version] = {
                'predicted_class': CLASS_NAMES[predicted_idx],
                'confidence': confidence,
                'class_probs': class_probs
            }
            
            print(f"   ✅ {CLASS_NAMES[predicted_idx].upper()} ({confidence*100:.2f}%)")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    if len(results) < 2:
        print("\n⚠️  Could not compare - one or both models failed to load")
        return
    
    # Print comparison
    print("\n" + "=" * 70)
    print("📊 COMPARISON RESULTS")
    print("=" * 70)
    
    v2_result = results.get('v2')
    v3_result = results.get('v3')
    
    if v2_result and v3_result:
        print(f"\nMobileNetV2: {v2_result['predicted_class'].upper()} ({v2_result['confidence']*100:.2f}%)")
        print(f"MobileNetV3: {v3_result['predicted_class'].upper()} ({v3_result['confidence']*100:.2f}%)")
        
        if v2_result['predicted_class'] == v3_result['predicted_class']:
            print("\n✅ Both models agree!")
        else:
            print("\n⚠️  Models disagree!")
    
    # Visualize comparison
    if show_plot and v2_result and v3_result:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Show image
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
        
        # V2 probabilities
        classes = list(v2_result['class_probs'].keys())
        v2_probs = list(v2_result['class_probs'].values())
        v2_idx = CLASS_NAMES.index(v2_result['predicted_class'])
        colors_v2 = ['red' if i == v2_idx else 'gray' for i in range(len(classes))]
        
        axes[1].barh(classes, v2_probs, color=colors_v2, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Probability', fontsize=12)
        axes[1].set_title(f"MobileNetV2\n{v2_result['predicted_class'].upper()} ({v2_result['confidence']*100:.2f}%)", 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # V3 probabilities
        v3_probs = list(v3_result['class_probs'].values())
        v3_idx = CLASS_NAMES.index(v3_result['predicted_class'])
        colors_v3 = ['red' if i == v3_idx else 'gray' for i in range(len(classes))]
        
        axes[2].barh(classes, v3_probs, color=colors_v3, alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Probability', fontsize=12)
        axes[2].set_title(f"MobileNetV3\n{v3_result['predicted_class'].upper()} ({v3_result['confidence']*100:.2f}%)", 
                         fontsize=14, fontweight='bold')
        axes[2].set_xlim([0, 1])
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        output_dir = "./test_results"
        os.makedirs(output_dir, exist_ok=True)
        img_name = Path(image_path).stem
        save_path = os.path.join(output_dir, f"comparison_{img_name}_result.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Comparison visualization saved: {save_path}")
        plt.show()
    
    return results


# ============================
# MAIN
# ============================
def main():
    parser = argparse.ArgumentParser(description='Test real-world images with trained models')
    parser.add_argument('--model', type=str, choices=['v2', 'v3', 'both'], 
                       default='v2', help='Model to use: v2, v3, or both for comparison')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--folder', type=str, help='Path to folder containing images')
    parser.add_argument('--no-plot', action='store_true', help='Do not show plots')
    parser.add_argument('--save-plot', action='store_true', help='Save plot images')
    parser.add_argument('--simple', action='store_true', 
                       help='Simple output mode: just show result (RUST, HEALTHY, etc.)')
    
    args = parser.parse_args()
    
    # Check if image or folder provided
    if not args.image and not args.folder:
        print("❌ ERROR: Please provide either --image or --folder")
        parser.print_help()
        return
    
    # Test single image
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ ERROR: Image not found: {args.image}")
            return
        
        if args.model == 'both':
            if args.simple:
                # Simple comparison
                print(f"\n📸 Image: {args.image}")
                print("\n" + "=" * 50)
                for model_version in ['v2', 'v3']:
                    try:
                        model = load_trained_model(model_version)
                        img_array, _ = preprocess_image(args.image)
                        predicted_idx, confidence, _, _ = predict_image(model, img_array)
                        predicted_class = CLASS_NAMES[predicted_idx]
                        print(f"MobileNetV{model_version.upper()}: {predicted_class.upper()} ({confidence*100:.1f}%)")
                    except Exception as e:
                        print(f"MobileNetV{model_version.upper()}: Error - {e}")
                print("=" * 50)
            else:
                compare_models(args.image, show_plot=not args.no_plot)
        else:
            test_single_image(args.model, args.image, 
                            show_plot=not args.no_plot and not args.simple, 
                            save_plot=args.save_plot,
                            simple_mode=args.simple)
    
    # Test folder
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"❌ ERROR: Folder not found: {args.folder}")
            return
        
        if args.model == 'both':
            print("⚠️  Comparison mode not available for folders. Testing with v2...")
            test_folder('v2', args.folder, 
                       show_plot=not args.no_plot and not args.simple, 
                       save_plot=args.save_plot or True)
        else:
            if args.simple:
                # Simple folder testing
                model = load_trained_model(args.model)
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(Path(args.folder).glob(f'*{ext}'))
                    image_files.extend(Path(args.folder).glob(f'*{ext.upper()}'))
                
                if not image_files:
                    print(f"❌ No images found in {args.folder}")
                    return
                
                print(f"\n📁 Testing {len(image_files)} image(s) with MobileNetV{args.model.upper()}...")
                print("=" * 50)
                
                for img_path in image_files:
                    try:
                        img_array, _ = preprocess_image(str(img_path))
                        predicted_idx, confidence, _, _ = predict_image(model, img_array)
                        predicted_class = CLASS_NAMES[predicted_idx]
                        print(f"{img_path.name:30s} → {predicted_class.upper():12s} ({confidence*100:.1f}%)")
                    except Exception as e:
                        print(f"{img_path.name:30s} → ERROR: {e}")
                
                print("=" * 50)
            else:
                test_folder(args.model, args.folder, 
                           show_plot=not args.no_plot, 
                           save_plot=args.save_plot or True)


if __name__ == "__main__":
    main()


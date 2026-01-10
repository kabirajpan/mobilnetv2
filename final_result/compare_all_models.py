"""
Model Comparison Script
=======================
Compares all trained models and generates comprehensive comparison reports
with MAP plots and performance metrics.

Usage:
    python compare_all_models.py
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

# ============================
# CONFIGURATION
# ============================
ROOT_DIR = ".."
MODEL_FOLDERS = ["mobilenetv2", "mobilenetv3", "resnet", "efficientnet"]
RESULTS_DIR = "./comparison_results"
PLOTS_DIR = "./comparison_plots"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 70)
print("🔍 MODEL COMPARISON ANALYSIS")
print("=" * 70)
print(f"📁 Root directory: {ROOT_DIR}")
print(f"📊 Results will be saved to: {RESULTS_DIR}")
print(f"📈 Plots will be saved to: {PLOTS_DIR}")
print("=" * 70)

# ============================
# LOAD RESULTS FROM ALL MODELS
# ============================
def load_model_results(model_folder):
    """Load the most recent results JSON file from a model folder"""
    results_path = os.path.join(ROOT_DIR, model_folder, "results")
    
    if not os.path.exists(results_path):
        return None
    
    # Find all JSON result files
    json_files = glob.glob(os.path.join(results_path, "results_*.json"))
    
    if not json_files:
        return None
    
    # Get the most recent file
    latest_file = max(json_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r') as f:
            results = json.load(f)
        results['model_folder'] = model_folder
        results['results_file'] = latest_file
        return results
    except Exception as e:
        print(f"⚠️  Error loading {latest_file}: {e}")
        return None


print("\n📂 Loading model results...")
all_results = {}

for model_folder in MODEL_FOLDERS:
    print(f"   Checking {model_folder}...", end=" ")
    results = load_model_results(model_folder)
    if results:
        all_results[model_folder] = results
        print(f"✅ Found results")
    else:
        print(f"❌ No results found")

if not all_results:
    print("\n❌ ERROR: No model results found!")
    print("Please train at least one model first.")
    exit(1)

print(f"\n✅ Loaded results from {len(all_results)} model(s)")

# ============================
# EXTRACT METRICS
# ============================
def extract_metrics(results_dict):
    """Extract key metrics from results dictionary"""
    metrics = {}
    
    for model_name, results in results_dict.items():
        metrics[model_name] = {
            'model': results.get('model', model_name),
            'val_accuracy': results.get('val_accuracy', 0),
            'tta_accuracy': results.get('tta_accuracy', 0),
            'train_accuracy': results.get('train_accuracy', 0),
            'map_score': results.get('map_score', 0),
            'macro_precision': results.get('macro_precision', 0),
            'macro_recall': results.get('macro_recall', 0),
            'macro_f1': results.get('macro_f1', 0),
            'weighted_precision': results.get('weighted_precision', 0),
            'weighted_recall': results.get('weighted_recall', 0),
            'weighted_f1': results.get('weighted_f1', 0),
            'generalization_gap': results.get('generalization_gap', 0),
            'total_time_hours': results.get('total_time_hours', 0),
            'training_samples': results.get('training_samples', 0),
            'validation_samples': results.get('validation_samples', 0),
        }
        
        # Add per-class AP if available
        if 'ap_per_class' in results:
            metrics[model_name]['ap_per_class'] = results['ap_per_class']
    
    return metrics


metrics = extract_metrics(all_results)

# ============================
# CREATE COMPARISON DATAFRAME
# ============================
comparison_data = []
for model_name, metric in metrics.items():
    comparison_data.append({
        'Model': metric['model'],
        'Validation Accuracy': metric['val_accuracy'] * 100,
        'TTA Accuracy': metric['tta_accuracy'] * 100,
        'MAP Score': metric['map_score'] * 100,
        'Macro F1-Score': metric['macro_f1'] * 100,
        'Macro Precision': metric['macro_precision'] * 100,
        'Macro Recall': metric['macro_recall'] * 100,
        'Weighted F1-Score': metric['weighted_f1'] * 100,
        'Generalization Gap': metric['generalization_gap'] * 100,
        'Training Time (hours)': metric['total_time_hours'],
    })

df = pd.DataFrame(comparison_data)
df = df.sort_values('TTA Accuracy', ascending=False)

# Save comparison table
comparison_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
df.to_csv(comparison_csv, index=False)
print(f"\n💾 Comparison table saved: {comparison_csv}")

# ============================
# PRINT COMPARISON SUMMARY
# ============================
print("\n" + "=" * 70)
print("📊 MODEL COMPARISON SUMMARY")
print("=" * 70)
print(df.to_string(index=False))
print("=" * 70)

# Find best model
best_model = df.iloc[0]
print(f"\n🏆 BEST MODEL: {best_model['Model']}")
print(f"   TTA Accuracy: {best_model['TTA Accuracy']:.2f}%")
print(f"   MAP Score: {best_model['MAP Score']:.2f}%")
print(f"   Macro F1-Score: {best_model['Macro F1-Score']:.2f}%")

# Find worst model (if more than one)
if len(df) > 1:
    worst_model = df.iloc[-1]
    print(f"\n⚠️  WORST MODEL: {worst_model['Model']}")
    print(f"   TTA Accuracy: {worst_model['TTA Accuracy']:.2f}%")
    print(f"   MAP Score: {worst_model['MAP Score']:.2f}%")
    print(f"   Macro F1-Score: {worst_model['Macro F1-Score']:.2f}%")

# ============================
# CREATE COMPARISON PLOTS
# ============================
print("\n📊 Creating comparison plots...")

# 1. Main Comparison Plot (4 subplots)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

models = df['Model'].values
x_pos = np.arange(len(models))

# Plot 1: Accuracy Comparison
axes[0, 0].bar(x_pos - 0.2, df['Validation Accuracy'], 0.4, 
               label='Validation Accuracy', alpha=0.8, color='skyblue')
axes[0, 0].bar(x_pos + 0.2, df['TTA Accuracy'], 0.4, 
               label='TTA Accuracy', alpha=0.8, color='lightcoral')
axes[0, 0].set_xlabel('Models', fontsize=12)
axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].set_ylim([0, 100])

# Add value labels on bars
for i, (val, tta) in enumerate(zip(df['Validation Accuracy'], df['TTA Accuracy'])):
    axes[0, 0].text(i - 0.2, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    axes[0, 0].text(i + 0.2, tta + 1, f'{tta:.1f}%', ha='center', va='bottom', fontsize=9)

# Plot 2: MAP Score Comparison
axes[0, 1].bar(x_pos, df['MAP Score'], alpha=0.8, color='mediumseagreen')
axes[0, 1].set_xlabel('Models', fontsize=12)
axes[0, 1].set_ylabel('MAP Score (%)', fontsize=12)
axes[0, 1].set_title('MAP (Mean Average Precision) Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].set_ylim([0, 100])

# Add value labels
for i, val in enumerate(df['MAP Score']):
    axes[0, 1].text(i, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: F1-Score Comparison
axes[1, 0].bar(x_pos - 0.2, df['Macro F1-Score'], 0.4, 
               label='Macro F1', alpha=0.8, color='plum')
axes[1, 0].bar(x_pos + 0.2, df['Weighted F1-Score'], 0.4, 
               label='Weighted F1', alpha=0.8, color='orange')
axes[1, 0].set_xlabel('Models', fontsize=12)
axes[1, 0].set_ylabel('F1-Score (%)', fontsize=12)
axes[1, 0].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].set_ylim([0, 100])

# Plot 4: Training Time vs Accuracy
scatter = axes[1, 1].scatter(df['Training Time (hours)'], df['TTA Accuracy'], 
                             s=200, alpha=0.6, c=range(len(df)), cmap='viridis')
for i, model in enumerate(models):
    axes[1, 1].annotate(model, 
                       (df['Training Time (hours)'].iloc[i], df['TTA Accuracy'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
axes[1, 1].set_xlabel('Training Time (hours)', fontsize=12)
axes[1, 1].set_ylabel('TTA Accuracy (%)', fontsize=12)
axes[1, 1].set_title('Training Time vs Accuracy', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
main_plot_file = os.path.join(PLOTS_DIR, "model_comparison_main.png")
plt.savefig(main_plot_file, dpi=300, bbox_inches='tight')
print(f"✅ Main comparison plot saved: {main_plot_file}")
plt.close()

# 2. MAP Comparison Plot (Detailed)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# MAP Bar Chart
axes[0].bar(x_pos, df['MAP Score'], alpha=0.8, color='mediumseagreen', edgecolor='black', linewidth=1.5)
axes[0].set_xlabel('Models', fontsize=12)
axes[0].set_ylabel('MAP Score (%)', fontsize=12)
axes[0].set_title('MAP (Mean Average Precision) Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(models, rotation=45, ha='right')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([0, 100])
axes[0].axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95% Target')
axes[0].legend()

# Add value labels
for i, val in enumerate(df['MAP Score']):
    axes[0].text(i, val + 1, f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Color code: green if >= 95%, yellow if >= 90%, red otherwise
    if val >= 95:
        color = 'green'
    elif val >= 90:
        color = 'orange'
    else:
        color = 'red'
    axes[0].bar(i, val, alpha=0.8, color=color, edgecolor='black', linewidth=1.5)

# Per-Class AP Comparison (if available)
if len(all_results) > 0:
    # Get class names from first model
    first_model = list(all_results.values())[0]
    if 'classes' in first_model and 'ap_per_class' in first_model:
        classes = first_model['classes']
        
        # Prepare data for per-class AP
        ap_data = []
        for model_name, results in all_results.items():
            if 'ap_per_class' in results:
                for class_name, ap in results['ap_per_class'].items():
                    ap_data.append({
                        'Model': results.get('model', model_name),
                        'Class': class_name,
                        'AP': ap * 100
                    })
        
        if ap_data:
            ap_df = pd.DataFrame(ap_data)
            
            # Create grouped bar chart
            x = np.arange(len(classes))
            width = 0.8 / len(models)
            
            for i, model in enumerate(models):
                model_ap = []
                for cls in classes:
                    model_data = ap_df[(ap_df['Model'] == model) & (ap_df['Class'] == cls)]
                    if not model_data.empty:
                        model_ap.append(model_data['AP'].values[0])
                    else:
                        model_ap.append(0)
                
                axes[1].bar(x + i * width, model_ap, width, label=model, alpha=0.8)
            
            axes[1].set_xlabel('Classes', fontsize=12)
            axes[1].set_ylabel('Average Precision (AP) (%)', fontsize=12)
            axes[1].set_title('Per-Class Average Precision (AP) Comparison', fontsize=14, fontweight='bold')
            axes[1].set_xticks(x + width * (len(models) - 1) / 2)
            axes[1].set_xticklabels(classes, rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].set_ylim([0, 100])
        else:
            axes[1].text(0.5, 0.5, 'Per-class AP data not available', 
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
            axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Per-class AP data not available', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].axis('off')
else:
    axes[1].axis('off')

plt.tight_layout()
map_plot_file = os.path.join(PLOTS_DIR, "map_comparison.png")
plt.savefig(map_plot_file, dpi=300, bbox_inches='tight')
print(f"✅ MAP comparison plot saved: {map_plot_file}")
plt.close()

# 3. Comprehensive Metrics Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Precision Comparison
axes[0, 0].bar(x_pos - 0.2, df['Macro Precision'], 0.4, 
               label='Macro Precision', alpha=0.8, color='steelblue')
axes[0, 0].bar(x_pos + 0.2, df['Weighted Precision'], 0.4, 
               label='Weighted Precision', alpha=0.8, color='lightblue')
axes[0, 0].set_xlabel('Models', fontsize=12)
axes[0, 0].set_ylabel('Precision (%)', fontsize=12)
axes[0, 0].set_title('Precision Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].set_ylim([0, 100])

# Recall Comparison
axes[0, 1].bar(x_pos, df['Macro Recall'], alpha=0.8, color='coral')
axes[0, 1].set_xlabel('Models', fontsize=12)
axes[0, 1].set_ylabel('Macro Recall (%)', fontsize=12)
axes[0, 1].set_title('Macro Recall Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].set_ylim([0, 100])

# Generalization Gap
axes[1, 0].bar(x_pos, df['Generalization Gap'], alpha=0.8, color='salmon')
axes[1, 0].set_xlabel('Models', fontsize=12)
axes[1, 0].set_ylabel('Generalization Gap (%)', fontsize=12)
axes[1, 0].set_title('Generalization Gap (Lower is Better)', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Overall Ranking
ranking_data = df[['Model', 'TTA Accuracy', 'MAP Score', 'Macro F1-Score']].copy()
ranking_data['Overall Score'] = (
    ranking_data['TTA Accuracy'] * 0.4 + 
    ranking_data['MAP Score'] * 0.4 + 
    ranking_data['Macro F1-Score'] * 0.2
)
ranking_data = ranking_data.sort_values('Overall Score', ascending=True)

axes[1, 1].barh(range(len(ranking_data)), ranking_data['Overall Score'], 
                alpha=0.8, color='mediumpurple')
axes[1, 1].set_yticks(range(len(ranking_data)))
axes[1, 1].set_yticklabels(ranking_data['Model'])
axes[1, 1].set_xlabel('Overall Score', fontsize=12)
axes[1, 1].set_title('Model Ranking (Weighted Score)', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
metrics_plot_file = os.path.join(PLOTS_DIR, "detailed_metrics_comparison.png")
plt.savefig(metrics_plot_file, dpi=300, bbox_inches='tight')
print(f"✅ Detailed metrics plot saved: {metrics_plot_file}")
plt.close()

# ============================
# GENERATE TEXT REPORT
# ============================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_file = os.path.join(RESULTS_DIR, f"comparison_report_{timestamp}.txt")

with open(report_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("MODEL COMPARISON REPORT\n")
    f.write("=" * 70 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Models Compared: {len(df)}\n\n")
    
    f.write("OVERALL RANKING:\n")
    f.write("-" * 70 + "\n")
    for idx, row in df.iterrows():
        f.write(f"{idx + 1}. {row['Model']}\n")
        f.write(f"   TTA Accuracy: {row['TTA Accuracy']:.2f}%\n")
        f.write(f"   MAP Score: {row['MAP Score']:.2f}%\n")
        f.write(f"   Macro F1-Score: {row['Macro F1-Score']:.2f}%\n")
        f.write(f"   Training Time: {row['Training Time (hours)']:.2f} hours\n")
        f.write("\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("BEST MODEL ANALYSIS\n")
    f.write("=" * 70 + "\n")
    f.write(f"Model: {best_model['Model']}\n")
    f.write(f"TTA Accuracy: {best_model['TTA Accuracy']:.2f}%\n")
    f.write(f"MAP Score: {best_model['MAP Score']:.2f}%\n")
    f.write(f"Macro F1-Score: {best_model['Macro F1-Score']:.2f}%\n")
    f.write(f"Training Time: {best_model['Training Time (hours)']:.2f} hours\n")
    
    if len(df) > 1:
        f.write("\n" + "=" * 70 + "\n")
        f.write("COMPARISON WITH OTHER MODELS\n")
        f.write("=" * 70 + "\n")
        for idx, row in df.iterrows():
            if row['Model'] != best_model['Model']:
                acc_diff = best_model['TTA Accuracy'] - row['TTA Accuracy']
                map_diff = best_model['MAP Score'] - row['MAP Score']
                f.write(f"\nvs {row['Model']}:\n")
                f.write(f"  Accuracy advantage: +{acc_diff:.2f}%\n")
                f.write(f"  MAP advantage: +{map_diff:.2f}%\n")

print(f"✅ Comparison report saved: {report_file}")

# ============================
# FINAL SUMMARY
# ============================
print("\n" + "=" * 70)
print("✅ COMPARISON COMPLETE!")
print("=" * 70)
print(f"📊 Comparison table: {comparison_csv}")
print(f"📈 Main comparison plot: {main_plot_file}")
print(f"📈 MAP comparison plot: {map_plot_file}")
print(f"📈 Detailed metrics plot: {metrics_plot_file}")
print(f"📄 Text report: {report_file}")
print("=" * 70)

print("\n🏆 BEST MODEL: " + best_model['Model'])
print(f"   TTA Accuracy: {best_model['TTA Accuracy']:.2f}%")
print(f"   MAP Score: {best_model['MAP Score']:.2f}%")
print(f"   Macro F1-Score: {best_model['Macro F1-Score']:.2f}%")

if best_model['TTA Accuracy'] >= 95:
    print("   ✅ EXCELLENT: Achieved 95%+ accuracy target!")
elif best_model['TTA Accuracy'] >= 90:
    print("   ✅ GOOD: Above 90% accuracy")
else:
    print("   ⚠️  Consider further training for better results")

print("\n" + "=" * 70)


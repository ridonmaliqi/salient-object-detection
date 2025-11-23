import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, jaccard_score
from data_loader import create_datasets
from sod_model import build_sod_model

# ---------------------------
# Config
# ---------------------------
DATASET_PATH = "./DUTS"
BASELINE_MODEL_PATH = "baseline_sod_model.keras"
IMPROVED_MODEL_PATH = "best_sod_model.keras"
BATCH_SIZE = 8
THRESHOLD = 0.5
CSV_FILE = "compare.csv"

# ---------------------------
# Load dataset
# ---------------------------
_, _, test_ds = create_datasets(DATASET_PATH)
imgs, masks = [], []
for img_batch, mask_batch in test_ds:
    imgs.append(img_batch.numpy())
    masks.append(mask_batch.numpy())
imgs = np.concatenate(imgs, axis=0)
masks = np.concatenate(masks, axis=0)

# ---------------------------
# Metric functions
# ---------------------------
def prf_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    return precision, recall, f1

def mae_metric(y_true, y_pred):
    return mean_absolute_error(y_true.flatten(), y_pred.flatten())

def iou_score(y_true, y_pred):
    return jaccard_score(y_true.flatten(), y_pred.flatten())

def evaluate_model(model, imgs, masks):
    preds = model.predict(imgs, batch_size=BATCH_SIZE)
    preds_bin = (preds > THRESHOLD).astype(np.uint8)
    masks_bin = (masks > 0.5).astype(np.uint8)
    
    ious = [iou_score(y_true, y_pred) for y_true, y_pred in zip(masks_bin, preds_bin)]
    precision, recall, f1 = prf_metrics(masks_bin, preds_bin)
    mae = mae_metric(masks, preds)
    
    return {
        "IoU": np.mean(ious),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MAE": mae,
        "preds_bin": preds_bin
    }

# ---------------------------
# Visualization function
# ---------------------------
def visualize_sample(img, mask, baseline_pred=None, improved_pred=None):
    img_disp = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    mask_disp = np.squeeze(mask)
    
    n_cols = 2 + (2 if baseline_pred is not None else 0)
    fig, axs = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
    
    axs[0].imshow(img_disp)
    axs[0].set_title("Input")
    axs[1].imshow(mask_disp, cmap='gray')
    axs[1].set_title("Ground Truth")
    
    if baseline_pred is not None:
        baseline_disp = np.squeeze(baseline_pred) * 255
        overlay_baseline = (0.5*img_disp + 0.5*np.stack([baseline_disp]*3, axis=-1)).astype(np.uint8)
        axs[2].imshow(baseline_disp, cmap='gray')
        axs[2].set_title("Baseline Pred")
        axs[3].imshow(overlay_baseline)
        axs[3].set_title("Overlay Baseline")
    
    plt.axis('off')
    plt.show()

# ---------------------------
# Load and evaluate models
# ---------------------------
baseline_results = None
if os.path.exists(BASELINE_MODEL_PATH):
    try:
        print("🔹 Loading baseline model...")
        baseline_model = build_sod_model()
        baseline_model.load_weights(BASELINE_MODEL_PATH)
        baseline_results = evaluate_model(baseline_model, imgs, masks)
        print("✅ Baseline model evaluated successfully.")
    except Exception as e:
        print(f"⚠️ Could not load baseline model: {e}")

print("🔹 Loading improved model...")
improved_model = build_sod_model()
improved_model.load_weights(IMPROVED_MODEL_PATH)
improved_results = evaluate_model(improved_model, imgs, masks)
print("✅ Improved model evaluated successfully.")

# ---------------------------
# Show visualizations (first 5 samples)
# ---------------------------
for i in range(min(5, len(imgs))):
    visualize_sample(
        imgs[i],
        masks[i],
        baseline_pred=baseline_results["preds_bin"][i] if baseline_results else None,
        improved_pred=improved_results["preds_bin"][i]
    )

# ---------------------------
# Prepare comparison table
# ---------------------------
metrics_keys = ["IoU", "Precision", "Recall", "F1", "MAE"]
comparison_data = []

for key in metrics_keys:
    row = [key]
    if baseline_results:
        row.append(round(baseline_results[key], 4))
    else:
        row.append("N/A")
    row.append(round(improved_results[key], 4))
    comparison_data.append(row)

# ---------------------------
# Save to CSV
# ---------------------------
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["Metric", "Baseline", "Improved"] if baseline_results else ["Metric", "Improved"]
    writer.writerow(header)
    writer.writerows(comparison_data)

print(f"\n✅ Comparison table saved to {CSV_FILE}")

# ---------------------------
# Print table
# ---------------------------
print("\n=== Metrics Comparison ===")
if baseline_results:
    print("{:<10} {:<10} {:<10}".format("Metric", "Baseline", "Improved"))
    for row in comparison_data:
        print("{:<10} {:<10} {:<10}".format(*row))
else:
    print("{:<10} {:<10}".format("Metric", "Improved"))
    for row in comparison_data:
        print("{:<10} {:<10}".format(row[0], row[1]))

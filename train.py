import os
import tensorflow as tf
from tensorflow.keras import optimizers
from sod_model import build_sod_model
from data_loader import create_datasets

# ---------------------------
# Config
# ---------------------------
DATASET_PATH = "DUTS"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 20
PATIENCE = 6
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = "best_sod_model.keras"
LOG_DIR = "logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------
# Prepare data
# ---------------------------
train_ds, val_ds, test_ds = create_datasets(DATASET_PATH)

# ---------------------------
# Loss & metrics
# ---------------------------
def iou_metric_batch(y_true, y_pred, smooth=1e-6):
    y_pred = tf.round(y_pred)
    y_true_f = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
    y_pred_f = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    union = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)

bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
def bce_iou_loss(y_true, y_pred):
    bce = bce_loss_fn(y_true, y_pred)
    iou = iou_metric_batch(y_true, y_pred)
    return bce + 0.5 * (1.0 - iou)

# ---------------------------
# Build model & compile
# ---------------------------
model = build_sod_model(input_shape=(128, 128, 3))
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=bce_iou_loss, metrics=[iou_metric_batch])

# ---------------------------
# Callbacks
# ---------------------------
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, "sod_model_epoch{epoch:02d}.weights.h5"),
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
    verbose=1
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=PATIENCE, restore_best_weights=True
)

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

# ---------------------------
# Training
# ---------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb, tensorboard_cb]
)

# ---------------------------
# Save final model
# ---------------------------
model.save(BEST_MODEL_PATH)
print(f"[SAVED] Best model at {BEST_MODEL_PATH}")

# ---------------------------
# Evaluate on test set
# ---------------------------
print("\n=== Evaluating on Test Set ===")
test_metrics = model.evaluate(test_ds)
print("Test metrics:", test_metrics)

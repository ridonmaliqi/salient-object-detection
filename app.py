import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
from sod_model import build_sod_model
import time

MODEL_PATH = "best_sod_model.keras"
IMG_SIZE = (128, 128)
THRESHOLD = 0.5

# Load model
model = build_sod_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
model.load_weights(MODEL_PATH)
model.trainable = False

def predict_saliency(img):
    img_array = np.array(img.resize(IMG_SIZE)) / 255.0
    img_array = img_array[np.newaxis, ...]
    start = time.time()
    pred = model.predict(img_array)
    end = time.time()
    mask_array = (pred[0, ..., 0] > THRESHOLD).astype(np.uint8) * 255
    overlay = (0.5 * np.array(img.resize(IMG_SIZE)) + 0.5 * np.stack([mask_array]*3, axis=-1)).astype(np.uint8)
    return mask_array, overlay, f"{end-start:.4f} sec"

demo = gr.Interface(
    fn=predict_saliency,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="numpy", label="Saliency Mask"),
        gr.Image(type="numpy", label="Overlay"),
        gr.Textbox(label="Inference Time")
    ],
    title="SOD Model Demo"
)

demo.launch()

# Salient Object Detection (SOD) from Scratch

This project implements a Salient Object Detection (SOD) system from scratch using TensorFlow/Keras. The model detects the most visually important object(s) in an image and outputs a saliency mask overlayed on the original image. No pre-trained models were used.

---

## Project Structure

```
project_root/
│
├─ data_loader.py       # Dataset loading, preprocessing, and augmentation
├─ sod_model.py         # CNN model architecture
├─ train.py             # Training and validation loop
├─ evaluate.py          # Evaluation metrics and visualization
├─ app.py               # Gradio demo for testing the model
├─ checkpoints/         # Saved model checkpoints (not pushed to GitHub)
├─ requirements.txt     # Python dependencies
├─ README.md
└─ .gitignore
```

### Dataset

This project uses the **DUTS** dataset. The dataset is **not included** in this repository due to size. You need to download it manually:

1. Download DUTS from the official website: [DUTS Dataset](http://saliencydetection.net/duts/)
2. Extract the zip file and structure your folder like this:

```
DUTS/
├─ images/  # Input images
└─ masks/   # Ground-truth saliency masks
```

3. Place the `DUTS` folder in the project root directory.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/salient-object-detection.git
cd salient-object-detection
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run the training script:

```bash
python train.py
```

The model will train for 15–25 epochs with early stopping based on validation loss. Checkpoints will be saved in the `checkpoints/` directory. The best-performing model will be saved as `best_sod_model.keras`.

---

## Evaluation

After training, evaluate your model using:

```bash
python evaluate.py
```

Metrics computed include:

- Intersection over Union (IoU)
- Precision, Recall, F1-Score
- Mean Absolute Error (MAE, optional)

Visualizations include:

- Input image
- Ground-truth mask
- Predicted mask
- Overlay (predicted + input)

---

## Demo

A Gradio demo is provided for easy interaction:

```bash
python app.py
```

The demo allows you to:

- Upload an image
- Display the predicted saliency mask
- Display an overlayed output
- Measure inference time per image

---

## Requirements

See `requirements.txt`:

```
tensorflow==2.15.0
numpy==1.25.2
matplotlib==3.8.0
opencv-python==4.9.0.76
scikit-learn==1.3.0
tqdm==4.66.1
gradio==3.45.1
```

---

## Notes

- Ensure the DUTS dataset folder is correctly structured as described above.
- Use a GPU if available for faster training and inference (Tesla T4 recommended on Colab).
- The project uses a custom CNN built from scratch; no pre-trained weights are used.
- For reproducibility, checkpoints save the model weights, optimizer state, and current epoch.

---

## References

- DUTS Dataset: [http://saliencydetection.net/duts/](http://saliencydetection.net/duts/)
- Salient Object Detection papers and CNN architectures.

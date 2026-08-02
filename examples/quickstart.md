# Quickstart

This walks through the full pipeline end-to-end: setup, training,
evaluation, and single-image prediction.

## 1. Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Prepare data

Download the 4-class brain MRI dataset (glioma / meningioma / notumor /
pituitary) and place it as:

```
data/raw/Training/<class_name>/*.jpg
data/raw/Testing/<class_name>/*.jpg
```

Or point `configs/config.yaml`'s `data.train_dir` / `data.test_dir` at
wherever your copy lives.

## 3. Train

```bash
python scripts/train.py --config configs/config.yaml
```

This will:
- discover class names from `data/raw/Training/`
- split off 15% of the training data for validation (stratified, seeded)
- train the VGG16-based model for the configured number of epochs
- save the best checkpoint to `artifacts/checkpoints/best_model.keras`
- save a full training-history CSV to `artifacts/logs/training_history.csv`

To do a quick smoke-test run with fewer epochs:

```bash
python scripts/train.py --config configs/config.yaml --epochs 2
```

## 4. Evaluate

```bash
python scripts/evaluate.py --config configs/config.yaml \
    --model-path artifacts/checkpoints/best_model.keras
```

Prints a classification report and per-class ROC-AUC to the console, and
saves a confusion matrix plot to `artifacts/evaluation/confusion_matrix.png`.

## 5. Predict on a single image

```bash
python scripts/predict.py --config configs/config.yaml \
    --model-path artifacts/checkpoints/best_model.keras \
    --image path/to/some_scan.jpg
```

Example output:

```
Tumor: glioma (confidence: 96.42%)

Per-class probabilities:
       glioma: 96.42%
   meningioma:  2.10%
      notumor:  0.88%
    pituitary:  0.60%
```

## 6. Programmatic usage in your own code

```python
from brain_tumor_classifier.config import load_config
from brain_tumor_classifier.data.dataset import discover_class_names
from brain_tumor_classifier.inference.predict import predict_image
import tensorflow as tf

config = load_config("configs/config.yaml")
class_names = discover_class_names(config.data.train_dir)
model = tf.keras.models.load_model("artifacts/checkpoints/best_model.keras")

result = predict_image(
    "path/to/scan.jpg", model, class_names, image_size=config.data.image_size
)

if result.is_tumor:
    print(f"Tumor detected: {result.predicted_class} ({result.confidence:.1%})")
else:
    print(f"No tumor detected ({result.confidence:.1%} confidence)")
```

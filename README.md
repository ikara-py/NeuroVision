# NeuroVision AI

Brain tumor classification from MRI images with a convolutional neural network (CNN).

NeuroVision AI classifies a brain MRI into four categories:

| Class | Description |
|---|---|
| `glioma` | Glioma tumor |
| `meningioma` | Meningioma tumor |
| `pituitary` | Pituitary tumor |
| `notumor` | No tumor |

> **Disclaimer:** this project is a decision-support tool built for educational purposes. It is not a medical device and never replaces the judgment of a physician.

## Project description

Interpreting a brain MRI takes time and specialized expertise, while the number of exams keeps growing. This project builds a model that can help sort exams and give a second opinion. It covers the full workflow:

- data discovery, cleaning and preparation (OpenCV, NumPy)
- design, training and evaluation of a multi-class CNN (TensorFlow / Keras)
- error analysis (confusion matrix, classification report, examples of mistakes)
- model saving and a prediction interface (Streamlit)

## Dataset

The dataset is available on Kaggle: [`karapy/neurovision`](https://www.kaggle.com/datasets/karapy/neurovision). It contains one folder per class (the folder name is the class label):

```
data/raw/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

Loading rules:

- only `.jpeg`, `.jpg`, `.bmp` and `.png` files are kept
- unreadable or corrupted images are skipped with a `try/except`, without stopping the loading
- all images are resized to **224 × 224** with OpenCV

## Project structure

```
NeuroVision/
├── data/
│   ├── raw/                   # original images, one folder per class
│   ├── processed/             # prepared data
│   └── excluded/              # files rejected during cleaning
├── models/
│   └── neurovision_model.keras   # trained model
├── notebooks/
│   └── cnn_classification.ipynb  # main notebook: data, model, training, evaluation
├── src/
│   └── neurovision/           # reusable Python package (loading functions, etc.)
├── testing_zone/              # experiments and drafts
├── .python-version
├── pyproject.toml             # dependencies
└── README.md
```

## Method

**Data preparation**

- stratified split: 80% train, 10% validation, 10% test (`random_state=42`)
- labels encoded with `LabelEncoder` (integer labels)
- pixel values normalized to the range [0, 1]
- data augmentation (light rotations, zoom, shifts) on the training set only

**Model**

| Layer | Details |
|---|---|
| Input | 224 × 224 × 3 |
| Conv blocks | 4 × (`Conv2D` 3×3, ReLU, same padding + `MaxPooling2D` 2×2) with 32, 64, 128 and 256 filters |
| Pooling | `GlobalAveragePooling2D` |
| Dense | 128 units, ReLU |
| Regularization | `Dropout(0.4)` |
| Output | `Dense(4)`, softmax |

- **Loss:** `sparse_categorical_crossentropy` (integer labels, 4 classes)
- **Optimizer:** Adam
- **Metric:** accuracy

**Evaluation**

The test set is used only once, at the end. The notebook produces:

- the test accuracy
- a confusion matrix
- a classification report (precision, recall, F1-score per class)

## How to run

### 1. Install

Requires the Python version given in `.python-version`.

```bash
git clone https://github.com/ikara-py/NeuroVision
cd NeuroVision
pip install -e .
```
### 2. Get the data

Download the dataset from Kaggle (`karapy/neurovision`) and place the four class folders in `data/raw/`. It can also be downloaded in Python:

```python
import kagglehub
path = kagglehub.dataset_download("karapy/neurovision")
```

### 3. Train and evaluate

Open `notebooks/cnn_classification.ipynb` and run all cells. The notebook trains the CNN, evaluates it on the test set and saves the model to `neurovision_model.keras`. Put this file in `models/`.

### 4. Predict on one image

```python
import tensorflow as tf

model = tf.keras.models.load_model("models/neurovision_model.keras")
```

Images must be preprocessed like in training: resized to 224 × 224 and divided by 255.

## Project management

Tasks are planned in Jira as Epics and tickets: Data, Model, Training, Evaluation, Deployment, Documentation.

## Tech stack

Python, TensorFlow / Keras, OpenCV, NumPy, scikit-learn, Matplotlib.

%pip install matplotlib numpy scikit-learn opencv-python-headless kagglehub

# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay


# ==========================================
# 2. DOWNLOAD AND CONFIGURE THE DATASET
# ==========================================


import kagglehub

# Download latest version
path = kagglehub.dataset_download("karapy/neurovision")

print("Path to dataset files:", path)

# DATA = Path('/kaggle/input/datasets/karapy/neurovision/Data/')

DATA = Path("/root/.cache/kagglehub/datasets/karapy/neurovision/versions/1/Data")

# Target image dimensions: 224 pixels high and 224 pixels wide

SIZE = (224, 224)

# Accepted image file extensions

EXTS = {'.jpeg', '.jpg', '.bmp', '.png'}

# Fixed random seed to make data splitting and shuffling reproducible

RANDOM_SEED = 42


# Number of images processed in each batch during training

BATCH = 32


# ==========================================
# 3. IDENTIFY THE IMAGE CLASSES
# ==========================================

# Each subdirectory inside DATA represents an MRI classification category
arr_names = []
for folder in DATA.iterdir():
    arr_names.append(folder.name)

arr_names

# ==========================================
# 4. COLLECT IMAGE PATHS AND LABELS
# ==========================================


def load_dataset(data_dir=DATA, arr_names=arr_names):

    # Store file paths instead of loading all image pixels into memory

    imgs_path_str, labels = [], []


    for class_name in arr_names:

        image_path = data_dir / class_name

        for path in image_path.iterdir():

            if path.suffix.lower() not in EXTS:
                continue

            imgs_path_str.append(str(path))
            labels.append(class_name)

    # Return NumPy arrays containing paths and their corresponding labels
    
    return np.array(imgs_path_str), np.array(labels)


# Collect image paths and class labels

imgs_path, labels = load_dataset()


# Check the number of collected paths and labels

print(imgs_path.shape, labels.shape)

# Inspect one image path and its corresponding class

print(imgs_path[0])
print(labels[0])

# Count how many images belong to each class

print(np.unique(labels, return_counts=True))

# ==========================================
# 5. ENCODE THE CLASS LABELS
# ==========================================


# Neural networks work with numerical targets rather than text labels

le = LabelEncoder()
labels_int = le.fit_transform(labels)

# Display the mapping between integer labels and original class names

print("Class mapping:", dict(enumerate(le.classes_)))
print(labels_int[:5])
 [markdown]
# ## Split data

# ==========================================
# 6. SPLIT THE DATASET
# ==========================================

# Reserve 80% for training and 20% for temporary data

X_train, X_temp, y_train, y_temp = train_test_split( imgs_path, labels_int, 
                                                            test_size=0.2, 
                                                            stratify=labels_int, 
                                                            random_state=RANDOM_SEED )

# Divide the temporary 20% equally:
# 10% for validation and 10% for testing

X_val, X_test, y_val, y_test = train_test_split( X_temp, y_temp, 
                                                        test_size=0.5, 
                                                        stratify=y_temp, 
                                                        random_state=RANDOM_SEED )

# Inspect the sizes of the three datasets

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# ==========================================
# 7. CREATE TF.DATA DATASETS
# ==========================================

# Create TensorFlow datasets from image paths and integer labels

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))


# ==========================================
# 8. LOAD AND PREPROCESS IMAGES
# ==========================================


def load_img(path, label):
    img = tf.io.read_file(path)
 
    img = tf.io.decode_image( img, 
                             channels=3, 
                             expand_animations=False )

    # resize the images
    img = tf.image.resize(img, SIZE)

    # normalize pixel values
    img = tf.keras.applications.vgg16.preprocess_input(img)

    return img, label

# ==========================================
# 9. BUILD THE INPUT PIPELINE
# ==========================================

# Training pipeline: map() -> shuffle() -> batch() -> prefetch()

train_ds = train_ds.map(load_img).shuffle(len(X_train), seed=RANDOM_SEED).batch(BATCH).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(load_img).batch(BATCH).prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(load_img).batch(BATCH).prefetch(tf.data.AUTOTUNE)

 [markdown]
# # **vgg16**

# ==========================================
# 10. LOAD THE PRETRAINED VGG MODEL
# ==========================================

base_model = tf.keras.applications.VGG16(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3))

base_model.trainable = False

# ==========================================
# 11. DEFINE TRAINING CALLBACKS
# ==========================================

early_stop = tf.keras.callbacks.EarlyStopping( monitor="val_loss", 
                                              patience=3, 
                                              restore_best_weights=True )

checkpoint = tf.keras.callbacks.ModelCheckpoint('vgg_mri.keras', save_best_only=True)


# ==========================================
# 12. BUILD THE CLASSIFICATION MODEL
# ==========================================


vgg_model = models.Sequential()
vgg_model.add(base_model)
vgg_model.add(tf.keras.layers.GlobalAveragePooling2D())
vgg_model.add(tf.keras.layers.Dense(128, activation='relu'))
vgg_model.add(tf.keras.layers.Dropout(0.5))
vgg_model.add(tf.keras.layers.Dense(4, activation='softmax'))

vgg_model.summary()


# ==========================================
# 13. COMPILE THE MODEL
# ==========================================

vgg_model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])

# ==========================================
# 14. TRAIN THE MODEL
# ==========================================


vgg_model_history = vgg_model.fit( train_ds, 
                                  epochs=10, 
                                  validation_data=val_ds, 
                                  verbose=2, 
                                  callbacks=[early_stop, checkpoint] )

# ==========================================
# 15. MAKE PREDICTIONS ON THE TEST DATASET
# ==========================================

vgg_pred = vgg_model.predict(test_ds)
 [markdown]
# ## Fine-tuning 

checkpoint = tf.keras.callbacks.ModelCheckpoint('fine_tunnung_vgg_model_mri.keras', save_best_only=True)

# ==========================================
# 16. PREPARE THE MODEL FOR FINE-TUNING
# ==========================================

base_model.trainable = True

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False


# ==========================================
# 17. KEEP BATCH NORMALIZATION LAYERS FROZEN
# ==========================================


for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# ==========================================
# 18. RECOMPILE THE MODEL FOR FINE-TUNING
# ==========================================

vgg_model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])


# ==========================================
# 19. CONTINUE TRAINING WITH FINE-TUNING
# ==========================================

vgg_model_history_finetune = vgg_model.fit( train_ds, 
                                           epochs=10, 
                                           validation_data=val_ds, 
                                           callbacks=[early_stop, checkpoint], 
                                           verbose=2 )
 [markdown]
# ## PLots

# ==========================================
# 20. PLOTS
# ==========================================


# Transfer Learning with VGG plots 


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Accuracy across epochs
axes[0].plot(vgg_model_history.history['accuracy'], label='Training accuracy')
axes[0].plot(vgg_model_history.history['val_accuracy'], label='Validation accuracy')
axes[0].set_title('Accuracy across epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# 2. Loss across epochs
axes[1].plot(vgg_model_history.history['loss'], label='Training loss')
axes[1].plot(vgg_model_history.history['val_loss'], label='Validation loss')
axes[1].set_title('Loss across epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)


# Adjust spacing to prevent overlapping titles and labels
plt.tight_layout()
plt.show()

# fine-tunning with VGG plots

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(vgg_model_history_finetune.history['accuracy'], label='Training accuracy')
axes[0].plot(vgg_model_history_finetune.history['val_accuracy'], label='Validation accuracy')
axes[0].set_title('Accuracy across epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(vgg_model_history_finetune.history['loss'], label='Training loss')
axes[1].plot(vgg_model_history_finetune.history['val_loss'], label='Validation loss')
axes[1].set_title('Loss across epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(vgg_model_history_finetune.history['accuracy'], label='Training accuracy')
axes[2].plot(vgg_model_history_finetune.history['loss'], label='Training loss')
axes[2].set_title('Training accuracy vs loss')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Value')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()



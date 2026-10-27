
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import kagglehub

# Download latest version
path = kagglehub.dataset_download("karapy/neurovision")

print("Path to dataset files:", path)

DATA = Path('/kaggle/input/datasets/karapy/neurovision/Data/')

# DATA = Path('/content/drive/MyDrive/NeuroVision/Data/raw/')
SIZE = (224, 224)
EXTS = {'.jpeg', '.jpg', '.bmp', '.png'}
RANDOM_SEED = 42

# get classes names

arr_names = []
for folder in DATA.iterdir():
    arr_names.append(folder.name)

arr_names


def load_dataset(data_dir=DATA, arr_names=arr_names):

    imgs_path_str, labels = [], []

    for class_name in arr_names:

        image_path = data_dir / class_name

        for path in image_path.iterdir():

            if path.suffix.lower() not in EXTS:
                continue

            imgs_path_str.append(str(path))
            labels.append(class_name)

    return np.array(imgs_path_str), np.array(labels)


imgs_path, labels = load_dataset()

print(imgs_path.shape, labels.shape)
print(imgs_path[0])
print(labels[0])
print(np.unique(labels, return_counts=True))

le = LabelEncoder()
labels_int = le.fit_transform(labels)

print("Class mapping:", dict(enumerate(le.classes_)))
print(labels_int[:5])

X_train, X_temp, y_train, y_temp = train_test_split( imgs_path, labels_int, 
                                                            test_size=0.2, 
                                                            stratify=labels_int, 
                                                            random_state=RANDOM_SEED )

X_val, X_test, y_val, y_test = train_test_split( X_temp, y_temp, 
                                                        test_size=0.5, 
                                                        stratify=y_temp, 
                                                        random_state=RANDOM_SEED )

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

def load_img(path, label):
    img = tf.io.read_file(path)
 
    img = tf.io.decode_image( img, 
                             channels=3, 
                             expand_animations=False )

    # resize the img
    img = tf.image.resize(img, SIZE)

    # normalize pixel values
    img = img / 255.0

    return img, label

train_ds = train_ds.map(load_img).shuffle(len(X_train), seed=RANDOM_SEED).batch(32).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(load_img).batch(32).prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(load_img).batch(32).prefetch(tf.data.AUTOTUNE)

img, label = next(iter(train_ds))

print("Image shape:", img.shape)
print("Image dtype:", img.dtype)
print("Label:", label.numpy())

# class distrbution

classes, counts = np.unique(labels, return_counts=True)

plt.figure(figsize=(7, 4))
bars = plt.bar(classes, counts, color=['blue', 'yellow', 'red', 'green'])
plt.bar_label(bars)
plt.title("Images per class")
plt.ylabel("Number of images")
plt.show()

my_model = models.Sequential()

my_model.add(tf.keras.Input((224,224, 3)))

my_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu', strides=1))
my_model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

my_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", strides=1))
my_model.add(tf.keras.layers.MaxPooling2D((2, 2)))

my_model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", strides=1))
my_model.add(tf.keras.layers.MaxPooling2D((2, 2)))

my_model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", strides=1))
my_model.add(tf.keras.layers.MaxPooling2D((2, 2)))

my_model.add(tf.keras.layers.GlobalAveragePooling2D())

my_model.add(tf.keras.layers.Dense(units=128, activation='relu'))

my_model.add(tf.keras.layers.Dense(units=4, activation='softmax'))

my_model.summary()

my_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping( monitor="val_loss", 
                                              patience=3, 
                                              restore_best_weights=True )

history = my_model.fit( train_ds, 
                       epochs=20, 
                       validation_data=val_ds, 
                       verbose=2, callbacks=[early_stop] )

test_loss, test_accuracy = my_model.evaluate(test_ds, verbose=2)

print(f"Test accuracy: {test_accuracy:.2%}")
print(f"Test loss: {test_loss:.4f}")

y_test[0]

model_pred = my_model.predict(test_ds)

pred_labels = np.argmax(model_pred, axis=1)

np.unique(pred_labels)

test_img = cv2.imread(X_test[0])
test_label = le.inverse_transform([y_test[0]])[0]
plt.imshow(test_img)
plt.title(f'y_true : {test_label}')

pred_label = le.inverse_transform([pred_labels[0]])[0]
plt.imshow(test_img)
plt.title(f'y_pred : {pred_label}')

cm = confusion_matrix(y_test, pred_labels)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=arr_names).plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion matrix (test set)")
plt.show()

print(classification_report(y_test, pred_labels, target_names=arr_names, digits=3))

my_model.save("neurovision_model.keras")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['accuracy'], label='Training accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation accuracy')
axes[0].set_title('Accuracy across epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'], label='Training loss')
axes[1].plot(history.history['val_loss'], label='Validation loss')
axes[1].set_title('Loss across epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(history.history['accuracy'], label='Training accuracy')
axes[2].plot(history.history['loss'], label='Training loss')
axes[2].set_title('Training accuracy vs loss')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Value')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
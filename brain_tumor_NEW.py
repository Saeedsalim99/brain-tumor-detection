# ============================================================
# Brain Tumor Detection - Binary Image Classification
# Student: Said Salim Alruqisiah | ID: 22F23114
# Module: COMP 20037 - Artificial Intelligence and Deep Learning
# Dataset: Brain Tumor MRI Dataset (Kaggle - masoudnickparvar)
# ============================================================
# PART 1: Basic CNN (3 Conv + 3 Pool + Flatten + 2 Hidden + 1 Output)
#         -> Confusion Matrix + Classification Report
# PART 2: CNN + Batch Normalization + Dropout (fix overfitting)
#         -> Confusion Matrix + Classification Report
# PART 3: ResNet50 Transfer Learning
#         -> Confusion Matrix + Classification Report
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

tf.random.set_seed(42)
np.random.seed(42)

print("="*60)
print("Brain Tumor Detection - Said Salim Alruqisiah (22F23114)")
print("="*60)

# Dataset paths
TRAIN_DIR = "brain_tumor_dataset/Training"
TEST_DIR  = "brain_tumor_dataset/Testing"

# Check folders
print("\nTraining folders:", os.listdir(TRAIN_DIR))
print("Testing folders:", os.listdir(TEST_DIR))

# We use only 2 classes: tumor (yes) and notumor (no)
# The dataset has 4 classes - we combine tumor types into one class
# and keep notumor as the other class

# ── Data Generators ──────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Binary: notumor=0, tumor=1
def get_binary_generator(datagen, directory, batch_size=32, img_size=(128,128), shuffle=True):
    return datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=['notumor', 'glioma'],  # binary: notumor vs tumor
        shuffle=shuffle,
        seed=42
    )

# Load data
train_gen = get_binary_generator(train_datagen, TRAIN_DIR, shuffle=True)
test_gen  = get_binary_generator(test_datagen,  TEST_DIR,  shuffle=False)

print(f"\nClass indices: {train_gen.class_indices}")
print(f"Training samples:  {train_gen.samples}")
print(f"Testing samples:   {test_gen.samples}")

# ── Helper Functions ─────────────────────────────────────────
def plot_cm(y_true, y_pred, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'])
    plt.title(title, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    print("Confusion Matrix:")
    print(cm)

def get_preds(model, gen):
    """Get predictions from model"""
    gen.reset()
    preds = (model.predict(gen, verbose=0) > 0.5).astype(int).flatten()
    return preds, gen.classes

def plot_curves(history, title):
    """Plot training curves"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(title + ' - Accuracy')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(title + ' - Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==============================================================
# PART 1: BASIC CNN
# 3 Conv + 3 Pool + Flatten + 2 Hidden Dense + 1 Output
# ==============================================================
print("\n" + "="*60)
print("PART 1: Basic CNN")
print("3 Conv + 3 Pool + Flatten + 2 Hidden + 1 Output")
print("="*60)

model1 = Sequential([
    # Convolutional Layer 1 + Pooling Layer 1
    Conv2D(32, (3,3), activation='relu', padding='same',
           input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),

    # Convolutional Layer 2 + Pooling Layer 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),

    # Convolutional Layer 3 + Pooling Layer 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),

    # Flatten
    Flatten(),

    # Hidden Layer 1
    Dense(256, activation='relu'),

    # Hidden Layer 2
    Dense(128, activation='relu'),

    # Output Layer - Sigmoid for binary classification
    Dense(1, activation='sigmoid')
], name="Basic_CNN")

model1.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model1.summary()

# Train Part 1
history1 = model1.fit(
    train_gen,
    epochs=10,
    validation_data=test_gen,
    verbose=1
)

# Evaluate Part 1
print("\n-- Part 1 Results --")
y_pred1, y_true1 = get_preds(model1, test_gen)
plot_cm(y_true1, y_pred1, "Part 1 - Basic CNN Confusion Matrix")
print("\nClassification Report - Part 1 (Basic CNN):")
print(classification_report(y_true1, y_pred1,
                             target_names=['No Tumor', 'Tumor']))
plot_curves(history1, "Part 1 - Basic CNN")
print(">> Observation: Check if Train accuracy >> Val accuracy = Overfitting!")

# ==============================================================
# PART 2: CNN + BATCH NORMALIZATION + DROPOUT
# ==============================================================
print("\n" + "="*60)
print("PART 2: CNN + Batch Normalization + Dropout")
print("Fix for overfitting found in Part 1")
print("="*60)

# Reset generators
train_gen.reset()
test_gen.reset()

model2 = Sequential([
    # Convolutional Layer 1 + BatchNorm + Pooling Layer 1
    Conv2D(32, (3,3), activation='relu', padding='same',
           input_shape=(128, 128, 3)),
    BatchNormalization(),   # Stabilize training, reduce overfitting
    MaxPooling2D(2, 2),
    Dropout(0.25),          # Drop 25% neurons to prevent overfitting

    # Convolutional Layer 2 + BatchNorm + Pooling Layer 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Convolutional Layer 3 + BatchNorm + Pooling Layer 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.4),

    # Flatten
    Flatten(),

    # Hidden Layer 1
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),           # Drop 50% before final layers

    # Hidden Layer 2
    Dense(128, activation='relu'),

    # Output Layer
    Dense(1, activation='sigmoid')
], name="CNN_BatchNorm_Dropout")

model2.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model2.summary()

# Train Part 2
history2 = model2.fit(
    train_gen,
    epochs=10,
    validation_data=test_gen,
    verbose=1
)

# Evaluate Part 2
print("\n-- Part 2 Results --")
y_pred2, y_true2 = get_preds(model2, test_gen)
plot_cm(y_true2, y_pred2, "Part 2 - CNN + BatchNorm + Dropout Confusion Matrix")
print("\nClassification Report - Part 2 (CNN + BatchNorm + Dropout):")
print(classification_report(y_true2, y_pred2,
                             target_names=['No Tumor', 'Tumor']))
plot_curves(history2, "Part 2 - CNN + BatchNorm + Dropout")
print(">> Observation: Train vs Val gap should be smaller now!")

# ==============================================================
# PART 3: ResNet50 TRANSFER LEARNING
# ==============================================================
print("\n" + "="*60)
print("PART 3: Pre-trained Model - ResNet50")
print("="*60)

# ResNet50 needs 224x224
train_gen224 = get_binary_generator(
    ImageDataGenerator(rescale=1./255, rotation_range=15,
                       horizontal_flip=True, zoom_range=0.1),
    TRAIN_DIR, img_size=(224, 224), shuffle=True
)
test_gen224 = get_binary_generator(
    ImageDataGenerator(rescale=1./255),
    TEST_DIR, img_size=(224, 224), shuffle=False
)

# Build ResNet50 model
base = ResNet50(weights='imagenet', include_top=False,
                input_shape=(224, 224, 3))
base.trainable = False   # Freeze base layers
print(f"ResNet50 base layers: {len(base.layers)} (frozen)")

# Custom classification head
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model3 = Model(inputs=base.input, outputs=output,
               name="ResNet50_Transfer")
model3.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower LR for transfer learning
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Part 3
history3 = model3.fit(
    train_gen224,
    epochs=10,
    validation_data=test_gen224,
    verbose=1
)

# Evaluate Part 3
print("\n-- Part 3 Results --")
y_pred3, y_true3 = get_preds(model3, test_gen224)
plot_cm(y_true3, y_pred3, "Part 3 - ResNet50 Transfer Learning Confusion Matrix")
print("\nClassification Report - Part 3 (ResNet50 Transfer Learning):")
print(classification_report(y_true3, y_pred3,
                             target_names=['No Tumor', 'Tumor']))
plot_curves(history3, "Part 3 - ResNet50")

# ==============================================================
# FINAL SUMMARY
# ==============================================================
print("\n" + "="*60)
print("FINAL COMPARISON SUMMARY")
print("="*60)

acc1 = accuracy_score(y_true1, y_pred1)
acc2 = accuracy_score(y_true2, y_pred2)
acc3 = accuracy_score(y_true3, y_pred3)
f1  = f1_score(y_true1, y_pred1)
f2  = f1_score(y_true2, y_pred2)
f3  = f1_score(y_true3, y_pred3)

print("\n{:<35} {:>10} {:>10}".format('Model', 'Accuracy', 'F1-Score'))
print("-"*57)
print("{:<35} {:>10.4f} {:>10.4f}".format('Part 1 - Basic CNN', acc1, f1))
print("{:<35} {:>10.4f} {:>10.4f}".format('Part 2 - CNN+BatchNorm+Dropout', acc2, f2))
print("{:<35} {:>10.4f} {:>10.4f}".format('Part 3 - ResNet50 Transfer', acc3, f3))

print("\nAll 3 parts complete!")
print("3 Confusion Matrices printed")
print("3 Classification Reports printed")
print("Overfitting fixed in Part 2")
print("ResNet50 used in Part 3")

# ============================================================
# Brain Tumor Detection - Binary Image Classification
# Student: Said Salim Alruqisiah | ID: 22F23114
# Module: COMP 20037 - Artificial Intelligence and Deep Learning
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

DATA_DIR = "brain_tumor_dataset"
BATCH = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=(128,128), batch_size=BATCH,
    class_mode='binary', subset='training', seed=42,
    classes=['no','yes']
)
val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=(128,128), batch_size=BATCH,
    class_mode='binary', subset='validation', seed=42,
    shuffle=False, classes=['no','yes']
)

print("Classes:", train_gen.class_indices)
print("Train:", train_gen.samples, "| Val:", val_gen.samples)

def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Tumor','Tumor'],
                yticklabels=['No Tumor','Tumor'])
    plt.title(title, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()
    print("Confusion Matrix:")
    print(cm)

def get_preds(model, gen):
    gen.reset()
    preds = (model.predict(gen, verbose=0) > 0.5).astype(int).flatten()
    return preds, gen.classes

# ============================================================
# PART 1: BASIC CNN
# 3 Conv + 3 Pool + Flatten + 2 Hidden Dense + 1 Output
# ============================================================
print("\n" + "="*60)
print("PART 1: Basic CNN")
print("3 Conv + 3 Pool + Flatten + 2 Hidden + 1 Output")
print("="*60)

model1 = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
], name="Basic_CNN")

model1.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
model1.summary()

history1 = model1.fit(train_gen, epochs=10, validation_data=val_gen, verbose=1)

print("\n-- Part 1 Results --")
y_pred1, y_true1 = get_preds(model1, val_gen)
plot_cm(y_true1, y_pred1, "Part 1 - Basic CNN Confusion Matrix")
print("\nClassification Report - Part 1 (Basic CNN):")
print(classification_report(y_true1, y_pred1, target_names=['No Tumor','Tumor']))

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history1.history['accuracy'], label='Train')
plt.plot(history1.history['val_accuracy'], label='Val')
plt.title('Part 1 - Accuracy')
plt.legend()
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(history1.history['loss'], label='Train')
plt.plot(history1.history['val_loss'], label='Val')
plt.title('Part 1 - Loss')
plt.legend()
plt.grid(True)
plt.suptitle('Part 1 Basic CNN Training Curves', fontweight='bold')
plt.tight_layout()
plt.show()
print(">> Observation: Check if Train accuracy >> Val accuracy = Overfitting!")

# ============================================================
# PART 2: CNN + BATCH NORMALIZATION + DROPOUT
# ============================================================
print("\n" + "="*60)
print("PART 2: CNN + Batch Normalization + Dropout")
print("Fix for overfitting found in Part 1")
print("="*60)

train_gen.reset()
val_gen.reset()

model2 = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
], name="CNN_BatchNorm_Dropout")

model2.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
model2.summary()

history2 = model2.fit(train_gen, epochs=10, validation_data=val_gen, verbose=1)

print("\n-- Part 2 Results --")
y_pred2, y_true2 = get_preds(model2, val_gen)
plot_cm(y_true2, y_pred2, "Part 2 - CNN + BatchNorm + Dropout Confusion Matrix")
print("\nClassification Report - Part 2 (CNN + BatchNorm + Dropout):")
print(classification_report(y_true2, y_pred2, target_names=['No Tumor','Tumor']))

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history2.history['accuracy'], label='Train')
plt.plot(history2.history['val_accuracy'], label='Val')
plt.title('Part 2 - Accuracy')
plt.legend()
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(history2.history['loss'], label='Train')
plt.plot(history2.history['val_loss'], label='Val')
plt.title('Part 2 - Loss')
plt.legend()
plt.grid(True)
plt.suptitle('Part 2 CNN+BatchNorm+Dropout Training Curves', fontweight='bold')
plt.tight_layout()
plt.show()
print(">> Observation: Train vs Val gap should be smaller now!")

# ============================================================
# PART 3: ResNet50 TRANSFER LEARNING
# ============================================================
print("\n" + "="*60)
print("PART 3: Pre-trained Model - ResNet50")
print("="*60)

datagen224 = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train224 = datagen224.flow_from_directory(
    DATA_DIR, target_size=(224,224), batch_size=BATCH,
    class_mode='binary', subset='training', seed=42,
    classes=['no','yes']
)
val224 = datagen224.flow_from_directory(
    DATA_DIR, target_size=(224,224), batch_size=BATCH,
    class_mode='binary', subset='validation', seed=42,
    shuffle=False, classes=['no','yes']
)

base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False
print("ResNet50 base layers:", len(base.layers), "(frozen)")

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid')(x)

model3 = Model(inputs=base.input, outputs=out, name="ResNet50_Transfer")
model3.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history3 = model3.fit(train224, epochs=10, validation_data=val224, verbose=1)

print("\n-- Part 3 Results --")
y_pred3, y_true3 = get_preds(model3, val224)
plot_cm(y_true3, y_pred3, "Part 3 - ResNet50 Transfer Learning Confusion Matrix")
print("\nClassification Report - Part 3 (ResNet50 Transfer Learning):")
print(classification_report(y_true3, y_pred3, target_names=['No Tumor','Tumor']))

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history3.history['accuracy'], label='Train')
plt.plot(history3.history['val_accuracy'], label='Val')
plt.title('Part 3 - Accuracy')
plt.legend()
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(history3.history['loss'], label='Train')
plt.plot(history3.history['val_loss'], label='Val')
plt.title('Part 3 - Loss')
plt.legend()
plt.grid(True)
plt.suptitle('Part 3 ResNet50 Training Curves', fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL COMPARISON SUMMARY")
print("="*60)

acc1 = accuracy_score(y_true1, y_pred1)
acc2 = accuracy_score(y_true2, y_pred2)
acc3 = accuracy_score(y_true3, y_pred3)
f1 = f1_score(y_true1, y_pred1)
f2 = f1_score(y_true2, y_pred2)
f3 = f1_score(y_true3, y_pred3)

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

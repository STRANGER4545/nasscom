import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Set paths to your train and test directories
train_dir = r"C:\Users\bitra\eye_disease_detection\dataset\train"
test_dir = r"C:\Users\bitra\eye_disease_detection\dataset\test"

# Preprocessing with ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.1, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=64, class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=64, class_mode='categorical'
)

# Load VGG19 Model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the 


checkpoint = ModelCheckpoint("../models/eye_disease_model.h5", save_best_only=True, monitor='val_accuracy', mode='max')
model.fit(train_generator, epochs=1, validation_data=test_generator, callbacks=[checkpoint], verbose=1)


# Save the model
os.makedirs("../models", exist_ok=True)
model.save("../models/eye_disease_model.h5")
print("[INFO] Model saved to models/eye_disease_model.h5")

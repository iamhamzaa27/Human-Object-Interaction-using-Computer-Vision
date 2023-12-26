import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

import os
from PIL import Image

def your_real_data_loader(batch_size):
    # Replace 'path/to/your/dataset' with the actual path to your dataset
    dataset_path = "D:\Semester 7\ML\Project\dataset\Train\True"

    # Get a list of image file names in the dataset
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg') or f.endswith('.png')]

    # Randomly select a batch of image file names
    batch_images = np.random.choice(image_files, size=batch_size, replace=False)

    # Load and preprocess the selected images
    images = []
    for image_file in batch_images:
        image_path = os.path.join(dataset_path, image_file)
        img = Image.open(image_path)
        img = img.resize((64, 64))  # Adjust the size as needed
        img_array = np.array(img) / 255.0  # Normalize pixel values to the range [0, 1]
        images.append(img_array)

    return np.array(images)

# Generator model
def build_generator(latent_dim):
    model = keras.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim, activation='relu'))
    model.add(layers.Reshape((8, 8, 4)))
    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='sigmoid'))
    return model

# Discriminator model
def build_discriminator(img_shape):
    model = keras.Sequential()
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=img_shape))
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Combined model
latent_dim = 100  # Adjust as needed
img_shape = (416, 416, 3)  # Adjust based on your dataset's image shape

generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)
gan = build_gan(generator, discriminator)

# Compile models
discriminator.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                      loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy', metrics=['accuracy'])

# Train GAN
batch_size = 64
epochs = 10  # Adjust based on your dataset and training needs

for epoch in range(epochs):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_images = generator.predict(noise)

    real_images = your_real_data_loader(batch_size)  # Replace with your real data loading function
    labels_real = np.ones((batch_size, 1))
    labels_fake = np.zeros((batch_size, 1))

    # Train discriminator on real data
    d_loss_real = discriminator.train_on_batch(real_images, labels_real)

    # Train discriminator on generated data
    d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    labels_gan = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, labels_gan)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss Real: {d_loss_real[0]}, D Loss Fake: {d_loss_fake[0]}, G Loss: {g_loss[0]}")

# Generate synthetic data
num_synthetic_samples = 5000  # Adjust as needed
synthetic_noise = np.random.normal(0, 1, (num_synthetic_samples, latent_dim))
synthetic_data = generator.predict(synthetic_noise)

# Save synthetic data or use it as needed

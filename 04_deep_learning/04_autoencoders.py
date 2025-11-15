"""
Autoencoders - Theory and Practice
Understanding autoencoders for dimensionality reduction and denoising.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist, fashion_mnist
import tensorflow as tf

print("=" * 70)
print("AUTOENCODERS - Theory & Practice")
print("=" * 70)

# ============================================================================
# THEORY: What are Autoencoders?
# ============================================================================
print("\n" + "=" * 70)
print("THEORY: Understanding Autoencoders")
print("=" * 70)

theory = """
AUTOENCODERS

1. WHAT ARE AUTOENCODERS?
   - Neural networks that learn to compress and reconstruct data
   - Unsupervised learning (no labels needed)
   - Two parts: Encoder (compress) and Decoder (reconstruct)

2. ARCHITECTURE:

   Input -> Encoder -> Latent Space -> Decoder -> Output
   
   - Encoder: Reduces dimensions (compression)
   - Latent Space: Compressed representation (bottleneck)
   - Decoder: Reconstructs from compressed representation

3. TYPES OF AUTOENCODERS:

   a) BASIC AUTOENCODER
      - Simple encoder-decoder
      - Learns to compress and reconstruct
   
   b) DENOISING AUTOENCODER
      - Trained on noisy data
      - Learns to remove noise
      - Input: noisy image, Output: clean image
   
   c) VARIATIONAL AUTOENCODER (VAE)
      - Learns probability distribution
      - Can generate new samples
      - Used in generative models
   
   d) SPARSE AUTOENCODER
      - Adds sparsity constraint
      - Fewer active neurons
      - Better feature learning

4. APPLICATIONS:
   - Dimensionality reduction
   - Image denoising
   - Anomaly detection
   - Feature learning
   - Data compression
   - Generative models (VAE)

5. HOW IT WORKS:
   - Encoder: x -> z (latent representation)
   - Decoder: z -> x' (reconstruction)
   - Loss: ||x - x'||² (reconstruction error)
   - Goal: Minimize reconstruction error
"""

print(theory)

# ============================================================================
# PRACTICE: Basic Autoencoder
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Building a Basic Autoencoder")
print("=" * 70)

# Load MNIST dataset
print("\nLoading MNIST dataset...")
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize and flatten
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(f"Training samples: {x_train.shape[0]}")
print(f"Input dimension: {x_train.shape[1]}")

# ============================================================================
# MODEL 1: Basic Autoencoder
# ============================================================================
print("\n" + "-" * 70)
print("Model 1: Basic Autoencoder")
print("-" * 70)

input_dim = 784  # 28x28
encoding_dim = 32  # Compressed representation

# Encoder
input_img = keras.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)

# Decoder
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

print("\nBasic Autoencoder Architecture:")
autoencoder.summary()

# Separate encoder and decoder
encoder = keras.Model(input_img, encoded)

encoded_input = keras.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

# ============================================================================
# MODEL 2: Deep Autoencoder
# ============================================================================
print("\n" + "-" * 70)
print("Model 2: Deep Autoencoder (Multiple Layers)")
print("-" * 70)

def build_deep_autoencoder(input_dim, encoding_dim):
    """Build a deep autoencoder"""
    input_img = keras.Input(shape=(input_dim,))
    
    # Encoder
    encoded = layers.Dense(128, activation='relu')(input_img)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = keras.Model(input_img, decoded)
    return autoencoder

deep_autoencoder = build_deep_autoencoder(input_dim, encoding_dim)
deep_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

print("\nDeep Autoencoder Architecture:")
deep_autoencoder.summary()

# ============================================================================
# MODEL 3: Convolutional Autoencoder (for images)
# ============================================================================
print("\n" + "-" * 70)
print("Model 3: Convolutional Autoencoder (Better for Images)")
print("-" * 70)

# Reshape for CNN
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

def build_conv_autoencoder():
    """Build a convolutional autoencoder"""
    input_img = keras.Input(shape=(28, 28, 1))
    
    # Encoder
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = keras.Model(input_img, decoded)
    return autoencoder

conv_autoencoder = build_conv_autoencoder()
conv_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

print("\nConvolutional Autoencoder Architecture:")
conv_autoencoder.summary()

# ============================================================================
# MODEL 4: Denoising Autoencoder
# ============================================================================
print("\n" + "-" * 70)
print("Model 4: Denoising Autoencoder")
print("-" * 70)

def add_noise(x, noise_factor=0.5):
    """Add noise to images"""
    noise = np.random.normal(0, noise_factor, x.shape)
    x_noisy = x + noise
    x_noisy = np.clip(x_noisy, 0., 1.)
    return x_noisy

# Create noisy training data
x_train_noisy = add_noise(x_train_cnn)
x_test_noisy = add_noise(x_test_cnn)

# Use same architecture as conv autoencoder
denoising_autoencoder = build_conv_autoencoder()
denoising_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

print("\nDenoising Autoencoder Architecture:")
print("(Same as Convolutional Autoencoder)")
print("Input: Noisy images")
print("Output: Clean images")

# ============================================================================
# TRAINING AND VISUALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("Training and Visualization")
print("=" * 70)

print("\nTo train and visualize, use this code:")
print("""
# Train basic autoencoder
history = autoencoder.fit(
    x_train, x_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=1
)

# Encode and decode test images
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Visualize
n = 10  # Number of digits to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.suptitle('Original (top) vs Reconstructed (bottom)')
plt.show()

# Visualize latent space
plt.figure(figsize=(10, 8))
scatter = plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], 
                     c=y_test, cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.title('Latent Space Visualization (2D)')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.show()
""")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. Autoencoders learn compressed representations
2. Encoder compresses, decoder reconstructs
3. Latent space contains learned features
4. Denoising autoencoders remove noise
5. Convolutional autoencoders work better for images
6. Can be used for anomaly detection

APPLICATIONS:
- Image compression
- Denoising
- Anomaly detection
- Feature learning
- Dimensionality reduction

NEXT STEPS:
- Try Variational Autoencoders (VAE)
- Use for anomaly detection
- Experiment with different architectures
- Try on your own images
"""

print(takeaways)

print("\n" + "=" * 70)
print("Next: Explore GANs for generative modeling")
print("=" * 70)


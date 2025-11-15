"""
Convolutional Neural Networks (CNNs) - Theory and Practice
Understanding CNNs for image classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10, mnist
import tensorflow as tf

print("=" * 70)
print("CONVOLUTIONAL NEURAL NETWORKS (CNNs) - Theory & Practice")
print("=" * 70)

# ============================================================================
# THEORY: What are CNNs?
# ============================================================================
print("\n" + "=" * 70)
print("THEORY: Understanding CNNs")
print("=" * 70)

theory = """
CONVOLUTIONAL NEURAL NETWORKS (CNNs)

1. WHY CNNs?
   - Traditional neural networks don't work well for images
   - Images have spatial structure (nearby pixels are related)
   - CNNs preserve and learn spatial relationships

2. KEY COMPONENTS:

   a) CONVOLUTION LAYER
      - Applies filters/kernels to detect features
      - Each filter learns to detect specific patterns (edges, shapes, etc.)
      - Creates feature maps
      
      Example: 3x3 filter detecting vertical edges
      Input:     Filter:     Output:
      [1 1 1]   [-1 0 1]   [0 0 0]
      [1 1 1] * [-1 0 1] = [0 0 0]
      [1 1 1]   [-1 0 1]   [0 0 0]
   
   b) POOLING LAYER
      - Reduces spatial dimensions
      - Max Pooling: Takes maximum value in each region
      - Average Pooling: Takes average value
      - Reduces computation and prevents overfitting
      
      Example: 2x2 Max Pooling
      Input:        Output:
      [1 3 2 1]     [3 2]
      [2 4 1 2]  -> [4 2]
      [3 1 2 3]
      [1 2 3 1]
   
   c) FULLY CONNECTED LAYER
      - Final classification layer
      - Takes flattened features
      - Outputs class probabilities

3. CNN ARCHITECTURE:
   Input Image
   -> Convolution + ReLU
   -> Pooling
   -> Convolution + ReLU
   -> Pooling
   -> Flatten
   -> Fully Connected
   -> Output (Classes)

4. KEY CONCEPTS:
   - Stride: How much filter moves (usually 1)
   - Padding: Adding zeros around image (same/valid)
   - Feature Maps: Output of convolution layers
   - Depth: Number of filters in a layer
"""

print(theory)

# ============================================================================
# PRACTICE: Building a Simple CNN
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Building a CNN for Image Classification")
print("=" * 70)

def build_simple_cnn(input_shape, num_classes):
    """Build a simple CNN architecture"""
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and Classify
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Load MNIST dataset (simpler for learning)
print("\nLoading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")
print(f"Image shape: {x_train.shape[1:]}")

# Build model
print("\nBuilding CNN model...")
model = build_simple_cnn(input_shape=(28, 28, 1), num_classes=10)

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
print("\nModel Architecture:")
model.summary()

# ============================================================================
# UNDERSTANDING THE ARCHITECTURE
# ============================================================================
print("\n" + "=" * 70)
print("Understanding Each Layer")
print("=" * 70)

layer_explanation = """
LAYER BREAKDOWN:

1. Conv2D(32, (3, 3))
   - 32 filters of size 3x3
   - Learns 32 different feature detectors
   - Output: 32 feature maps

2. MaxPooling2D((2, 2))
   - Reduces size by half (28x28 -> 14x14)
   - Takes max value in 2x2 regions
   - Reduces parameters, prevents overfitting

3. Conv2D(64, (3, 3))
   - 64 filters, learns more complex features
   - Builds on previous layer's features

4. Flatten()
   - Converts 2D feature maps to 1D vector
   - Prepares for fully connected layers

5. Dense(64)
   - Fully connected layer
   - Learns combinations of features

6. Dense(10, softmax)
   - Output layer
   - 10 classes (digits 0-9)
   - Softmax gives probabilities
"""

print(layer_explanation)

# ============================================================================
# TRAINING (Optional - commented out to save time)
# ============================================================================
print("\n" + "=" * 70)
print("Training the Model")
print("=" * 70)
print("\nTo train the model, uncomment the training code below:")
print("""
# Train model (this takes time)
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\\nTest accuracy: {test_acc:.4f}')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# VISUALIZING CONVOLUTIONS
# ============================================================================
print("\n" + "=" * 70)
print("Visualizing What CNNs Learn")
print("=" * 70)

def visualize_filters(model, layer_idx=0, num_filters=8):
    """Visualize learned filters"""
    print(f"\nVisualizing filters from layer {layer_idx}...")
    
    # Get weights from first convolutional layer
    layer = model.layers[layer_idx]
    filters, biases = layer.get_weights()
    
    # Normalize filters to 0-1 range
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    # Plot first few filters
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(min(num_filters, 8)):
        row, col = i // 4, i % 4
        axes[row, col].imshow(filters[:, :, 0, i], cmap='gray')
        axes[row, col].set_title(f'Filter {i+1}')
        axes[row, col].axis('off')
    
    plt.suptitle('Learned Filters (First Convolutional Layer)')
    plt.tight_layout()
    plt.show()

print("\nNote: To visualize filters, train the model first, then call:")
print("visualize_filters(model)")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. CNNs are designed for images with spatial structure
2. Convolution layers learn feature detectors
3. Pooling reduces size and prevents overfitting
4. Deeper layers learn more complex features
5. Transfer learning can speed up training
6. Data augmentation helps with limited data

NEXT STEPS:
- Try CIFAR-10 dataset (color images)
- Experiment with different architectures
- Try transfer learning (VGG, ResNet)
- Build your own image classifier
"""

print(takeaways)

print("\n" + "=" * 70)
print("Next: Try 03_transfer_learning.py for advanced CNN techniques")
print("=" * 70)


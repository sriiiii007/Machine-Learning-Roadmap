"""
Neural Network Basics - Building from Scratch
This file demonstrates the fundamentals of neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    """
    A simple neural network with one hidden layer.
    This is a learning example - in practice, use TensorFlow/Keras.
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward propagation"""
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """Backward propagation"""
        m = X.shape[0]  # Number of samples
        
        # Calculate gradients
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=1000):
        """Train the neural network"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss (Mean Squared Error)
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)


# Example: XOR Problem (classic neural network test)
if __name__ == "__main__":
    print("=" * 50)
    print("Neural Network Basics - XOR Problem")
    print("=" * 50)
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create and train network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
    
    print("\nTraining Neural Network...")
    losses = nn.train(X, y, epochs=2000)
    
    # Make predictions
    print("\nPredictions:")
    predictions = nn.predict(X)
    for i, (input_val, pred, actual) in enumerate(zip(X, predictions, y)):
        print(f"Input: {input_val} -> Prediction: {pred[0]:.4f} (Actual: {actual[0]})")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    print("\n" + "=" * 50)
    print("Next Steps:")
    print("1. Experiment with different learning rates")
    print("2. Try different activation functions (ReLU, Tanh)")
    print("3. Add more hidden layers")
    print("4. Move to TensorFlow/Keras for real projects")
    print("=" * 50)


"""
Recurrent Neural Networks (RNNs) and LSTMs - Theory and Practice
Understanding RNNs for sequence data and time series.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

print("=" * 70)
print("RECURRENT NEURAL NETWORKS (RNNs) & LSTMs - Theory & Practice")
print("=" * 70)

# ============================================================================
# THEORY: What are RNNs?
# ============================================================================
print("\n" + "=" * 70)
print("THEORY: Understanding RNNs and LSTMs")
print("=" * 70)

theory = """
RECURRENT NEURAL NETWORKS (RNNs)

1. WHY RNNs?
   - Traditional neural networks can't handle sequences
   - RNNs have memory - they remember previous inputs
   - Perfect for: time series, text, speech, sequences

2. HOW RNNs WORK:

   Basic RNN Cell:
   - Takes current input (x_t) and previous hidden state (h_{t-1})
   - Produces output (y_t) and new hidden state (h_t)
   - Hidden state acts as memory
   
   Formula:
   h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
   y_t = W_hy * h_t + b_y

3. THE VANISHING GRADIENT PROBLEM:
   - RNNs struggle with long sequences
   - Gradients become very small during backpropagation
   - Can't learn long-term dependencies
   - Solution: LSTMs and GRUs

4. LONG SHORT-TERM MEMORY (LSTM):
   - Solves vanishing gradient problem
   - Has three gates:
     a) Forget Gate: What to forget from memory
     b) Input Gate: What new information to store
     c) Output Gate: What to output
   - Can remember information for long periods

5. GATED RECURRENT UNIT (GRU):
   - Simpler than LSTM
   - Combines forget and input gates
   - Often performs similarly to LSTM
   - Faster to train

6. APPLICATIONS:
   - Text generation
   - Language translation
   - Sentiment analysis
   - Time series forecasting
   - Speech recognition
"""

print(theory)

# ============================================================================
# PRACTICE: Building an RNN for Sentiment Analysis
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Building RNN for Sentiment Analysis")
print("=" * 70)

# Load IMDB movie reviews dataset
print("\nLoading IMDB dataset...")
vocab_size = 10000
max_length = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to same length
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
print(f"Sequence length: {max_length}")

# ============================================================================
# MODEL 1: Simple RNN
# ============================================================================
print("\n" + "-" * 70)
print("Model 1: Simple RNN")
print("-" * 70)

def build_simple_rnn():
    """Build a simple RNN model"""
    model = keras.Sequential([
        layers.Embedding(vocab_size, 128, input_length=max_length),
        layers.SimpleRNN(64, return_sequences=False),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

rnn_model = build_simple_rnn()
rnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nSimple RNN Architecture:")
rnn_model.summary()

# ============================================================================
# MODEL 2: LSTM
# ============================================================================
print("\n" + "-" * 70)
print("Model 2: LSTM (Better for Long Sequences)")
print("-" * 70)

def build_lstm():
    """Build an LSTM model"""
    model = keras.Sequential([
        layers.Embedding(vocab_size, 128, input_length=max_length),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

lstm_model = build_lstm()
lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nLSTM Architecture:")
lstm_model.summary()

# ============================================================================
# MODEL 3: Bidirectional LSTM
# ============================================================================
print("\n" + "-" * 70)
print("Model 3: Bidirectional LSTM (Sees Both Directions)")
print("-" * 70)

def build_bidirectional_lstm():
    """Build a bidirectional LSTM model"""
    model = keras.Sequential([
        layers.Embedding(vocab_size, 128, input_length=max_length),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

bidirectional_model = build_bidirectional_lstm()
bidirectional_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nBidirectional LSTM Architecture:")
bidirectional_model.summary()

# ============================================================================
# UNDERSTANDING THE LAYERS
# ============================================================================
print("\n" + "=" * 70)
print("Understanding Each Layer")
print("=" * 70)

layer_explanation = """
LAYER BREAKDOWN:

1. Embedding Layer
   - Converts word indices to dense vectors
   - Learns word representations
   - Input: word indices (integers)
   - Output: dense vectors (embeddings)

2. LSTM Layer
   - Processes sequence one step at a time
   - Maintains hidden state (memory)
   - Learns long-term dependencies
   - return_sequences=False: only last output
   - return_sequences=True: all outputs

3. Bidirectional LSTM
   - Processes sequence in both directions
   - Forward: left to right
   - Backward: right to left
   - Better context understanding

4. Dropout
   - Prevents overfitting
   - Randomly sets some neurons to zero

5. Dense (Output)
   - Final classification
   - Sigmoid for binary classification
"""

print(layer_explanation)

# ============================================================================
# TRAINING (Optional)
# ============================================================================
print("\n" + "=" * 70)
print("Training the Models")
print("=" * 70)
print("\nTo train, uncomment the code below:")
print("""
# Train LSTM model
print("Training LSTM model...")
history = lstm_model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=3,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_acc = lstm_model.evaluate(x_test, y_test, verbose=0)
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
# TIME SERIES FORECASTING EXAMPLE
# ============================================================================
print("\n" + "=" * 70)
print("Time Series Forecasting with LSTM")
print("=" * 70)

def create_time_series_data():
    """Create synthetic time series data"""
    # Generate sine wave with trend
    time_steps = 1000
    t = np.linspace(0, 4*np.pi, time_steps)
    data = np.sin(t) + 0.1 * np.random.randn(time_steps) + 0.01 * t
    
    return data

def prepare_sequences(data, lookback=10):
    """Prepare sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

print("\nCreating synthetic time series data...")
ts_data = create_time_series_data()
lookback = 20
X_ts, y_ts = prepare_sequences(ts_data, lookback)

# Reshape for LSTM (samples, timesteps, features)
X_ts = X_ts.reshape(X_ts.shape[0], X_ts.shape[1], 1)

print(f"Time series shape: {X_ts.shape}")
print(f"Target shape: {y_ts.shape}")

# Build LSTM for time series
ts_model = keras.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
    layers.LSTM(50, return_sequences=False),
    layers.Dense(1)
])

ts_model.compile(optimizer='adam', loss='mse')

print("\nTime Series LSTM Model:")
ts_model.summary()

print("\nTo train time series model:")
print("""
history = ts_model.fit(
    X_ts[:800], y_ts[:800],
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    verbose=0
)

# Predict
predictions = ts_model.predict(X_ts[800:])

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_ts[800:], label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Time Series Forecasting with LSTM')
plt.legend()
plt.show()
""")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. RNNs are designed for sequential data
2. LSTMs solve the vanishing gradient problem
3. Bidirectional LSTMs see both directions
4. Embedding layer converts words to vectors
5. Use LSTM for text, time series, sequences
6. GRU is simpler alternative to LSTM

WHEN TO USE:
- Text classification: LSTM/Bidirectional LSTM
- Time series: LSTM
- Text generation: LSTM with return_sequences=True
- Simple sequences: SimpleRNN or GRU

NEXT STEPS:
- Try text generation with LSTM
- Build a time series predictor
- Experiment with GRU
- Move to Transformers for better performance
"""

print(takeaways)

print("\n" + "=" * 70)
print("Next: Explore Transformers for even better sequence modeling")
print("=" * 70)


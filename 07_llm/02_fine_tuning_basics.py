"""
Fine-tuning LLMs - Practical Guide
Learning to fine-tune pre-trained language models.
"""

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM
)
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("FINE-TUNING LLMs - Theory & Practice")
print("=" * 70)

# ============================================================================
# THEORY: Fine-tuning
# ============================================================================
print("\n" + "=" * 70)
print("THEORY: Understanding Fine-tuning")
print("=" * 70)

theory = """
FINE-TUNING LARGE LANGUAGE MODELS

1. WHAT IS FINE-TUNING?
   - Adapt pre-trained model to specific task
   - Train on your domain data
   - Much faster than training from scratch
   - Better performance than zero-shot

2. WHY FINE-TUNE?
   - Pre-trained models are general
   - Your task is specific
   - Better accuracy on your data
   - Domain adaptation

3. TYPES OF FINE-TUNING:

   a) FULL FINE-TUNING
      - Update all parameters
      - Best performance
      - Requires more resources
   
   b) PARAMETER-EFFICIENT FINE-TUNING (PEFT)
      - Update only some parameters
      - LoRA (Low-Rank Adaptation)
      - QLoRA (Quantized LoRA)
      - Faster, less memory

4. PROCESS:
   1. Load pre-trained model
   2. Prepare your dataset
   3. Add task-specific head (if needed)
   4. Train on your data
   5. Evaluate and use

5. CONSIDERATIONS:
   - Learning rate (usually very small)
   - Number of epochs (few is often enough)
   - Batch size (depends on GPU memory)
   - Data quality matters
"""

print(theory)

# ============================================================================
# PRACTICE: Fine-tuning for Classification
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Fine-tuning BERT for Text Classification")
print("=" * 70)

# Note: This is a demonstration. For actual training, you need:
# - GPU (recommended) or CPU (slower)
# - Sufficient memory
# - Training dataset

print("\nSetting up fine-tuning example...")
print("Note: This demonstrates the process. Full training requires GPU.")

# Sample data (in practice, load from file)
texts = [
    "I love this product!",
    "This is terrible.",
    "Amazing quality!",
    "Very disappointed.",
    "Great service!",
    "Poor experience.",
] * 10  # Repeat for more data

labels = [1, 0, 1, 0, 1, 0] * 10  # 1 = positive, 0 = negative

print(f"\nSample dataset:")
print(f"Texts: {len(texts)}")
print(f"Labels: {len(labels)}")
print(f"Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")

# Create dataset
dataset_dict = {
    'text': texts,
    'label': labels
}
dataset = Dataset.from_dict(dataset_dict)

# Split
train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test['train']
test_dataset = train_test['test']

print(f"\nTrain: {len(train_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")

# ============================================================================
# SETUP MODEL AND TOKENIZER
# ============================================================================
print("\n" + "=" * 70)
print("Setting up Model and Tokenizer")
print("=" * 70)

model_name = "distilbert-base-uncased"  # Smaller, faster model for demo
print(f"\nUsing model: {model_name}")
print("Note: For production, use larger models like 'bert-base-uncased'")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

# Tokenize datasets
print("\nTokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Load model
print("\nLoading pre-trained model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # Binary classification
)

print(f"Model loaded: {model_name}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# TRAINING SETUP
# ============================================================================
print("\n" + "=" * 70)
print("Training Configuration")
print("=" * 70)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,           # Few epochs for fine-tuning
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,            # Small learning rate
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

print("\nTraining configuration:")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Learning rate: {training_args.learning_rate}")
print(f"Batch size: {training_args.per_device_train_batch_size}")

# ============================================================================
# TRAINING (Commented out - requires GPU for real training)
# ============================================================================
print("\n" + "=" * 70)
print("Training Process")
print("=" * 70)

print("""
To actually train the model, uncomment and run:

trainer.train()

This will:
1. Train for specified epochs
2. Evaluate on test set each epoch
3. Save best model
4. Log training metrics

Training time depends on:
- Dataset size
- Model size
- GPU/CPU
- Number of epochs

For this small example: ~5-10 minutes on GPU
For larger datasets: Hours to days
""")

# ============================================================================
# INFERENCE EXAMPLE
# ============================================================================
print("\n" + "=" * 70)
print("Using Fine-tuned Model for Inference")
print("=" * 70)

print("""
After training, use the model like this:

# Load fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained('./results/best_model')
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare input
text = "I love this product!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = predictions.argmax().item()
    confidence = predictions.max().item()

print(f"Sentiment: {'Positive' if predicted_class == 1 else 'Negative'}")
print(f"Confidence: {confidence:.4f}")
""")

# ============================================================================
# PARAMETER-EFFICIENT FINE-TUNING (LoRA)
# ============================================================================
print("\n" + "=" * 70)
print("Parameter-Efficient Fine-tuning (LoRA)")
print("=" * 70)

print("""
LoRA (Low-Rank Adaptation) is a more efficient fine-tuning method:

Advantages:
- Updates only small number of parameters
- Much less memory required
- Faster training
- Can combine multiple LoRA adapters

Example using PEFT library:

from peft import LoraConfig, get_peft_model, TaskType

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                    # Rank
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Shows only small % trainable

# Train as before
trainer = Trainer(model=model, ...)
trainer.train()

Benefits:
- 10-100x fewer parameters to train
- Can train on consumer GPUs
- Multiple adapters for different tasks
""")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. Fine-tuning adapts pre-trained models to your task
2. Much faster than training from scratch
3. Requires labeled data for your task
4. Small learning rate is crucial
5. Few epochs often sufficient
6. LoRA is more efficient than full fine-tuning

BEST PRACTICES:
- Start with small learning rate (1e-5 to 5e-5)
- Use few epochs (1-5)
- Monitor validation performance
- Use appropriate batch size
- Consider LoRA for efficiency

WHEN TO FINE-TUNE:
- You have labeled data
- Pre-trained model doesn't perform well
- Domain-specific task
- Need better accuracy

NEXT STEPS:
- Try with your own dataset
- Experiment with hyperparameters
- Try LoRA for efficiency
- Explore advanced techniques
"""

print(takeaways)

print("\n" + "=" * 70)
print("Next: Explore advanced LLM techniques")
print("=" * 70)


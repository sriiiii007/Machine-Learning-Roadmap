"""
Working with Pre-trained LLMs
Introduction to using Hugging Face Transformers for LLM tasks.
"""

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline
)
import torch

def text_classification_example():
    """Example: Text classification using pre-trained model"""
    print("=" * 60)
    print("Text Classification with BERT")
    print("=" * 60)
    
    # Using pipeline (easiest way)
    classifier = pipeline("sentiment-analysis")
    
    texts = [
        "I love this product!",
        "This is terrible.",
        "It's okay, nothing special."
    ]
    
    print("\nSentiment Analysis Results:")
    for text, result in zip(texts, classifier(texts)):
        print(f"\nText: {text}")
        print(f"Label: {result['label']}")
        print(f"Score: {result['score']:.4f}")


def text_generation_example():
    """Example: Text generation using GPT-2"""
    print("\n" + "=" * 60)
    print("Text Generation with GPT-2")
    print("=" * 60)
    
    # Using pipeline
    generator = pipeline("text-generation", model="gpt2", max_length=100)
    
    prompt = "The future of artificial intelligence"
    
    print(f"\nPrompt: {prompt}")
    print("\nGenerated Text:")
    results = generator(prompt, num_return_sequences=1, max_length=100)
    print(results[0]['generated_text'])


def question_answering_example():
    """Example: Question answering"""
    print("\n" + "=" * 60)
    print("Question Answering")
    print("=" * 60)
    
    qa_pipeline = pipeline("question-answering")
    
    context = """
    Machine Learning is a subset of artificial intelligence that focuses on 
    algorithms that can learn from data. Deep Learning is a subset of machine 
    learning that uses neural networks with multiple layers.
    """
    
    question = "What is deep learning?"
    
    print(f"\nContext: {context.strip()}")
    print(f"\nQuestion: {question}")
    
    result = qa_pipeline(question=question, context=context)
    print(f"\nAnswer: {result['answer']}")
    print(f"Confidence: {result['score']:.4f}")


def using_models_directly():
    """Example: Using models directly (more control)"""
    print("\n" + "=" * 60)
    print("Using Models Directly (Advanced)")
    print("=" * 60)
    
    # Load tokenizer and model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Prepare input
    text = "I love machine learning!"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get labels (you'd need to know the label mapping)
    labels = ["NEGATIVE", "POSITIVE"]
    predicted_label = labels[predictions.argmax().item()]
    confidence = predictions.max().item()
    
    print(f"\nText: {text}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")


def prompt_engineering_basics():
    """Introduction to prompt engineering"""
    print("\n" + "=" * 60)
    print("Prompt Engineering Basics")
    print("=" * 60)
    
    generator = pipeline("text-generation", model="gpt2", max_length=150)
    
    prompts = [
        "Write a story about AI:",
        "Write a story about AI. Make it creative and engaging:",
        "You are a creative writer. Write a story about AI:"
    ]
    
    print("\nDifferent Prompts, Different Results:\n")
    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: {prompt}")
        result = generator(prompt, num_return_sequences=1, max_length=150)
        print(f"Result: {result[0]['generated_text'][:200]}...")
        print("-" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Working with Pre-trained LLMs")
    print("=" * 60)
    print("\nNote: First run will download models (may take time)")
    print("=" * 60)
    
    try:
        # Run examples
        text_classification_example()
        text_generation_example()
        question_answering_example()
        using_models_directly()
        prompt_engineering_basics()
        
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("1. Experiment with different models")
        print("2. Learn about fine-tuning")
        print("3. Explore advanced prompt engineering")
        print("4. Try LoRA for efficient fine-tuning")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have transformers installed:")
        print("pip install transformers torch")
        print("\nFor GPU support:")
        print("pip install torch --index-url https://download.pytorch.org/whl/cu118")


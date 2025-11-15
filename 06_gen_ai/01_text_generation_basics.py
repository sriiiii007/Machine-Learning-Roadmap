"""
Generative AI - Text Generation Basics
Introduction to text generation with language models.
"""

from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch


def simple_text_generation():
    """Simple text generation using pipeline"""
    print("=" * 60)
    print("Simple Text Generation")
    print("=" * 60)
    
    # Create text generation pipeline
    generator = pipeline("text-generation", model="gpt2", max_length=100)
    
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a world",
        "Machine learning is"
    ]
    
    print("\nGenerating text for different prompts:\n")
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        result = generator(prompt, num_return_sequences=1, max_length=100)
        generated = result[0]['generated_text']
        print(f"Generated: {generated[:200]}...")
        print("-" * 60)


def controlled_generation():
    """Controlled text generation with parameters"""
    print("\n" + "=" * 60)
    print("Controlled Text Generation")
    print("=" * 60)
    
    generator = pipeline("text-generation", model="gpt2")
    
    prompt = "The benefits of machine learning are"
    
    print(f"\nPrompt: {prompt}\n")
    
    # Different sampling strategies
    strategies = {
        "Conservative (low temperature)": {
            "temperature": 0.3,
            "top_k": 50,
            "top_p": 0.9
        },
        "Balanced": {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9
        },
        "Creative (high temperature)": {
            "temperature": 1.2,
            "top_k": 50,
            "top_p": 0.95
        }
    }
    
    for name, params in strategies.items():
        print(f"{name}:")
        result = generator(
            prompt,
            max_length=80,
            num_return_sequences=1,
            **params
        )
        print(f"  {result[0]['generated_text'][:150]}...")
        print()


def understanding_generation_parameters():
    """Explain generation parameters"""
    print("\n" + "=" * 60)
    print("Understanding Generation Parameters")
    print("=" * 60)
    
    explanation = """
    Key Parameters for Text Generation:
    
    1. TEMPERATURE (0.1 - 2.0)
       - Controls randomness
       - Low (0.1-0.5): More deterministic, focused
       - High (1.0-2.0): More creative, diverse
       - Default: 1.0
    
    2. TOP-K
       - Consider only top K most likely tokens
       - Reduces unlikely outputs
       - Typical: 50-100
    
    3. TOP-P (Nucleus Sampling)
       - Consider tokens with cumulative probability <= p
       - More dynamic than top-k
       - Typical: 0.9-0.95
    
    4. MAX_LENGTH
       - Maximum tokens to generate
       - Balance between completeness and cost
    
    5. NUM_RETURN_SEQUENCES
       - Number of different outputs to generate
       - Useful for exploring options
    """
    
    print(explanation)


def creative_writing_example():
    """Example: Creative writing assistant"""
    print("\n" + "=" * 60)
    print("Creative Writing Example")
    print("=" * 60)
    
    generator = pipeline("text-generation", model="gpt2", max_length=150)
    
    story_prompt = """Write a short story about a robot learning to paint:"""
    
    print(f"Prompt: {story_prompt}\n")
    
    result = generator(
        story_prompt,
        max_length=200,
        temperature=0.8,
        top_p=0.9,
        num_return_sequences=1
    )
    
    print("Generated Story:")
    print(result[0]['generated_text'])


def code_generation_example():
    """Example: Code generation (conceptual)"""
    print("\n" + "=" * 60)
    print("Code Generation (Conceptual)")
    print("=" * 60)
    
    print("""
    For code generation, you would use specialized models like:
    - CodeGPT
    - Codex (OpenAI)
    - StarCoder
    - CodeLlama
    
    Example usage:
    
    from transformers import pipeline
    
    code_generator = pipeline("text-generation", model="microsoft/CodeGPT-small-py")
    
    prompt = "def fibonacci(n):"
    code = code_generator(prompt, max_length=100)
    
    Note: GPT-2 used here is for general text, not optimized for code.
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Generative AI - Text Generation Basics")
    print("=" * 60)
    print("\nNote: First run will download GPT-2 model (may take time)")
    print("=" * 60)
    
    try:
        # Run examples
        simple_text_generation()
        controlled_generation()
        understanding_generation_parameters()
        creative_writing_example()
        code_generation_example()
        
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("1. Experiment with different models (GPT-2, GPT-Neo)")
        print("2. Try image generation (Stable Diffusion)")
        print("3. Explore multi-modal generation")
        print("4. Learn about GANs and VAEs")
        print("5. Fine-tune models for specific tasks")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nInstall required packages:")
        print("pip install transformers torch")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have internet connection for model download")


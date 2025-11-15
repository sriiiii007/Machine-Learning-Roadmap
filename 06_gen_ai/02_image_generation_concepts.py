"""
Image Generation Concepts - Understanding GANs and Diffusion Models
Theoretical and practical understanding of image generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("IMAGE GENERATION CONCEPTS - Theory & Practice")
print("=" * 70)

# ============================================================================
# THEORY: Image Generation
# ============================================================================
print("\n" + "=" * 70)
print("THEORY: Understanding Image Generation")
print("=" * 70)

theory = """
IMAGE GENERATION WITH AI

1. GENERATIVE ADVERSARIAL NETWORKS (GANs):

   Architecture:
   - Generator: Creates fake images
   - Discriminator: Distinguishes real from fake
   - Adversarial training: They compete
   
   How it works:
   1. Generator creates fake image from noise
   2. Discriminator tries to identify if real or fake
   3. Generator learns to fool discriminator
   4. Discriminator learns to detect fakes
   5. Both improve iteratively
   
   Types:
   - DCGAN: Deep Convolutional GAN
   - StyleGAN: High-quality face generation
   - CycleGAN: Image-to-image translation

2. DIFFUSION MODELS:

   How it works:
   1. Forward process: Add noise to image gradually
   2. Reverse process: Remove noise to generate image
   3. Model learns to denoise
   
   Advantages:
   - More stable training than GANs
   - High quality results
   - Better diversity
   
   Examples:
   - Stable Diffusion
   - DALL-E 2
   - Midjourney

3. VARIATIONAL AUTOENCODERS (VAEs):

   How it works:
   1. Encoder: Compress image to latent space
   2. Decoder: Reconstruct from latent space
   3. Sample from latent space to generate
   
   Use cases:
   - Image generation
   - Image editing
   - Anomaly detection
"""

print(theory)

# ============================================================================
# PRACTICE: Understanding GANs Conceptually
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: GAN Training Process (Conceptual)")
print("=" * 70)

# Simulate GAN training process
np.random.seed(42)

print("\nSimulating GAN training process...")

# Simulate discriminator accuracy over training
epochs = 20
discriminator_accuracy = []
generator_quality = []

for epoch in range(epochs):
    # Early: Discriminator wins
    if epoch < 5:
        disc_acc = 0.9 - epoch * 0.05
        gen_qual = 0.2 + epoch * 0.1
    # Middle: Competition
    elif epoch < 15:
        disc_acc = 0.65 + np.random.rand() * 0.1
        gen_qual = 0.7 + epoch * 0.02
    # Late: Generator improves
    else:
        disc_acc = 0.55 + np.random.rand() * 0.1
        gen_qual = 0.9 + (epoch - 15) * 0.02
    
    discriminator_accuracy.append(disc_acc)
    generator_quality.append(min(gen_qual, 0.98))

# Visualize training process
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(epochs), discriminator_accuracy, 'o-', label='Discriminator Accuracy', linewidth=2)
plt.plot(range(epochs), generator_quality, 's-', label='Generator Quality', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('GAN Training: Adversarial Process', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

plt.subplot(1, 2, 2)
# Show conceptual improvement
epochs_to_show = [0, 5, 10, 15, 19]
for i, epoch in enumerate(epochs_to_show):
    # Simulate image quality improvement
    quality = generator_quality[epoch]
    noise_level = 1 - quality
    
    # Create simple visualization
    img = np.random.rand(10, 10) * noise_level + (1 - noise_level) * 0.5
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Epoch {epoch}\nQuality: {quality:.2f}', fontsize=8)
    plt.axis('off')

plt.suptitle('Generator Improvement Over Time', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('gan_training_process.png', dpi=150, bbox_inches='tight')
print("\nSaved: gan_training_process.png")
plt.close()

print("\nGAN Training Observations:")
print("- Early: Discriminator easily distinguishes real from fake")
print("- Middle: Generator improves, competition intensifies")
print("- Late: Generator creates realistic images")

# ============================================================================
# PRACTICE: Diffusion Process Visualization
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Understanding Diffusion Process")
print("=" * 70)

# Simulate diffusion forward and reverse process
print("\nSimulating diffusion process...")

# Forward: Add noise
steps = 10
original = np.random.rand(20, 20)  # Simple "image"

forward_images = [original.copy()]
current = original.copy()

for step in range(1, steps + 1):
    # Add noise
    noise = np.random.randn(20, 20) * (step / steps)
    current = current + noise * 0.3
    forward_images.append(current.copy())

# Reverse: Remove noise (generation)
reverse_images = [forward_images[-1].copy()]  # Start from noisy
current = forward_images[-1].copy()

for step in range(steps - 1, -1, -1):
    # Remove noise (simplified)
    noise_estimate = np.random.randn(20, 20) * (step / steps) * 0.2
    current = current - noise_estimate
    reverse_images.append(current.copy())

# Visualize
fig, axes = plt.subplots(2, 6, figsize=(15, 5))

# Forward process
for i in range(6):
    idx = i * (len(forward_images) // 6)
    axes[0, i].imshow(forward_images[idx], cmap='gray')
    axes[0, i].set_title(f'Step {idx}', fontsize=8)
    axes[0, i].axis('off')
axes[0, 0].set_title('Original', fontsize=8)
axes[0, -1].set_title('Noisy', fontsize=8)

# Reverse process
for i in range(6):
    idx = (len(reverse_images) - 1) - i * (len(reverse_images) // 6)
    axes[1, i].imshow(reverse_images[idx], cmap='gray')
    axes[1, i].set_title(f'Step {len(reverse_images) - idx - 1}', fontsize=8)
    axes[1, i].axis('off')
axes[1, 0].set_title('Noisy', fontsize=8)
axes[1, -1].set_title('Generated', fontsize=8)

plt.suptitle('Diffusion: Forward (Add Noise) → Reverse (Generate)', fontweight='bold')
plt.tight_layout()
plt.savefig('diffusion_process.png', dpi=150, bbox_inches='tight')
print("\nSaved: diffusion_process.png")
plt.close()

print("\nDiffusion Process:")
print("- Forward: Gradually add noise to image")
print("- Reverse: Model learns to remove noise")
print("- Generation: Start from noise, denoise to create image")

# ============================================================================
# COMPARISON: GANs vs Diffusion
# ============================================================================
print("\n" + "=" * 70)
print("Comparison: GANs vs Diffusion Models")
print("=" * 70)

comparison_data = {
    'Aspect': ['Training Stability', 'Image Quality', 'Diversity', 'Training Time', 'Inference Speed', 'Use Case'],
    'GANs': ['Unstable', 'Very High', 'Good', 'Fast', 'Fast', 'Real-time generation'],
    'Diffusion': ['Stable', 'Very High', 'Excellent', 'Slow', 'Slow', 'High-quality generation']
}

import pandas as pd
comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# PRACTICAL USAGE
# ============================================================================
print("\n" + "=" * 70)
print("Practical Usage: Stable Diffusion")
print("=" * 70)

print("""
Using Stable Diffusion (example):

from diffusers import StableDiffusionPipeline
import torch

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")  # GPU recommended

# Generate image
prompt = "A beautiful sunset over mountains, photorealistic"
image = pipe(prompt).images[0]

# Save
image.save("generated_image.png")

Key Parameters:
- prompt: Text description
- num_inference_steps: More steps = better quality (slower)
- guidance_scale: How closely to follow prompt
- negative_prompt: What to avoid
""")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. GANs: Generator vs Discriminator competition
2. Diffusion: Add noise, then remove to generate
3. VAEs: Encode-decode with sampling
4. Each has different strengths

GANs:
- Fast generation
- Can be unstable
- Good for real-time

Diffusion:
- Very high quality
- Stable training
- Slower generation

VAEs:
- Good for editing
- Smooth latent space
- Lower quality than GANs/Diffusion

CHOOSING:
- Real-time: GANs
- Best quality: Diffusion
- Editing: VAEs
- Production: Pre-trained models (Stable Diffusion, DALL-E)

NEXT STEPS:
- Try Stable Diffusion
- Experiment with prompts
- Learn prompt engineering
- Explore image editing
"""

print(takeaways)

print("\n" + "=" * 70)
print("Next: Explore multimodal generation and advanced techniques")
print("=" * 70)


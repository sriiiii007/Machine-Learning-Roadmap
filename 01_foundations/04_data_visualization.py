"""
Data Visualization with Matplotlib and Seaborn
Essential visualization techniques for data analysis and ML.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 70)
print("DATA VISUALIZATION - Theory & Practice")
print("=" * 70)

# ============================================================================
# THEORY: Why Visualization?
# ============================================================================
print("\n" + "=" * 70)
print("THEORY: Understanding Data Visualization")
print("=" * 70)

theory = """
DATA VISUALIZATION

1. WHY VISUALIZE?
   - Understand data patterns
   - Detect outliers and anomalies
   - Communicate findings
   - Explore relationships
   - Validate assumptions

2. TYPES OF VISUALIZATIONS:

   a) UNIVARIATE (Single Variable)
      - Histogram: Distribution of values
      - Box plot: Distribution and outliers
      - Bar chart: Categorical counts
   
   b) BIVARIATE (Two Variables)
      - Scatter plot: Relationship between two variables
      - Line plot: Trends over time
      - Heatmap: Correlation matrix
   
   c) MULTIVARIATE (Multiple Variables)
      - Pair plot: All pairwise relationships
      - 3D scatter: Three variables
      - Faceted plots: Multiple subplots

3. TOOLS:
   - Matplotlib: Basic plotting
   - Seaborn: Statistical visualizations
   - Plotly: Interactive plots

4. BEST PRACTICES:
   - Clear labels and titles
   - Appropriate colors
   - Right chart type for data
   - Avoid clutter
"""

print(theory)

# ============================================================================
# PRACTICE: Basic Plots with Matplotlib
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Basic Plots")
print("=" * 70)

# Create sample data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)

# 1. Line Plot
print("\n1. Creating line plot...")
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Sine wave with noise', linewidth=2)
plt.plot(x, np.sin(x), 'r--', label='True sine wave', linewidth=2)
plt.xlabel('X values', fontsize=12)
plt.ylabel('Y values', fontsize=12)
plt.title('Line Plot Example', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('line_plot.png', dpi=150, bbox_inches='tight')
print("Saved: line_plot.png")
plt.close()

# 2. Scatter Plot
print("\n2. Creating scatter plot...")
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100) * 0.5

plt.figure(figsize=(10, 6))
plt.scatter(x_scatter, y_scatter, alpha=0.6, s=50, c='blue', edgecolors='black')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Scatter Plot - Relationship Between Variables', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_plot.png', dpi=150, bbox_inches='tight')
print("Saved: scatter_plot.png")
plt.close()

# 3. Histogram
print("\n3. Creating histogram...")
data_hist = np.random.normal(100, 15, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data_hist, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Histogram - Distribution of Data', fontsize=14, fontweight='bold')
plt.axvline(np.mean(data_hist), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data_hist):.2f}')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('histogram.png', dpi=150, bbox_inches='tight')
print("Saved: histogram.png")
plt.close()

# ============================================================================
# ADVANCED PLOTS
# ============================================================================
print("\n" + "=" * 70)
print("Advanced Plots")
print("=" * 70)

# Create sample dataset
np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.random.randn(200),
    'feature2': np.random.randn(200),
    'feature3': np.random.randn(200),
    'category': np.random.choice(['A', 'B', 'C'], 200),
    'target': np.random.choice([0, 1], 200)
})

# 4. Box Plot
print("\n4. Creating box plot...")
plt.figure(figsize=(10, 6))
df.boxplot(column=['feature1', 'feature2', 'feature3'])
plt.title('Box Plot - Distribution and Outliers', fontsize=14, fontweight='bold')
plt.ylabel('Value', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('box_plot.png', dpi=150, bbox_inches='tight')
print("Saved: box_plot.png")
plt.close()

# 5. Bar Chart
print("\n5. Creating bar chart...")
category_counts = df['category'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(category_counts.index, category_counts.values, color=['skyblue', 'lightcoral', 'lightgreen'], edgecolor='black')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Bar Chart - Category Distribution', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('bar_chart.png', dpi=150, bbox_inches='tight')
print("Saved: bar_chart.png")
plt.close()

# ============================================================================
# SEABORN PLOTS (Statistical Visualizations)
# ============================================================================
print("\n" + "=" * 70)
print("Seaborn Statistical Plots")
print("=" * 70)

# 6. Correlation Heatmap
print("\n6. Creating correlation heatmap...")
correlation_matrix = df[['feature1', 'feature2', 'feature3']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: heatmap.png")
plt.close()

# 7. Pair Plot
print("\n7. Creating pair plot...")
plt.figure(figsize=(12, 10))
sns.pairplot(df[['feature1', 'feature2', 'feature3', 'category']], hue='category', diag_kind='hist')
plt.suptitle('Pair Plot - All Pairwise Relationships', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('pair_plot.png', dpi=150, bbox_inches='tight')
print("Saved: pair_plot.png")
plt.close()

# 8. Violin Plot
print("\n8. Creating violin plot...")
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='category', y='feature1', inner='box')
plt.title('Violin Plot - Distribution by Category', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Feature 1', fontsize=12)
plt.tight_layout()
plt.savefig('violin_plot.png', dpi=150, bbox_inches='tight')
print("Saved: violin_plot.png")
plt.close()

# ============================================================================
# PRACTICAL ML VISUALIZATION EXAMPLE
# ============================================================================
print("\n" + "=" * 70)
print("Practical ML Visualization: EDA (Exploratory Data Analysis)")
print("=" * 70)

# Create realistic ML dataset
np.random.seed(42)
ml_data = pd.DataFrame({
    'age': np.random.randint(18, 65, 500),
    'income': np.random.normal(50000, 15000, 500),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 500),
    'purchased': np.random.choice([0, 1], 500, p=[0.6, 0.4])
})

print("\nCreating comprehensive EDA visualization...")

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Distribution of age
axes[0, 0].hist(ml_data['age'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].set_title('Age Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. Income by education
ml_data.boxplot(column='income', by='education', ax=axes[0, 1])
axes[0, 1].set_title('Income by Education Level', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Education')
axes[0, 1].set_ylabel('Income')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Purchase rate by education
purchase_rate = ml_data.groupby('education')['purchased'].mean()
axes[1, 0].bar(purchase_rate.index, purchase_rate.values, color='lightcoral', edgecolor='black')
axes[1, 0].set_title('Purchase Rate by Education', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Education')
axes[1, 0].set_ylabel('Purchase Rate')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Scatter: Age vs Income (colored by purchased)
scatter = axes[1, 1].scatter(ml_data['age'], ml_data['income'], 
                             c=ml_data['purchased'], cmap='RdYlGn', 
                             alpha=0.6, s=50, edgecolors='black')
axes[1, 1].set_title('Age vs Income (Color: Purchased)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Income')
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 1], label='Purchased')

plt.suptitle('Exploratory Data Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('eda_dashboard.png', dpi=150, bbox_inches='tight')
print("Saved: eda_dashboard.png")
plt.close()

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. Visualization is crucial for understanding data
2. Different plots for different purposes:
   - Histogram: Distribution
   - Scatter: Relationships
   - Box plot: Outliers and distribution
   - Heatmap: Correlations
3. Always label axes and add titles
4. Use appropriate colors and styles
5. EDA helps before building ML models

COMMON VISUALIZATIONS FOR ML:
- Distribution plots: Understand feature distributions
- Correlation heatmap: Find feature relationships
- Scatter plots: Visualize relationships
- Box plots: Detect outliers
- Pair plots: Comprehensive EDA

NEXT STEPS:
- Practice with real datasets
- Learn to interpret visualizations
- Use visualizations in your ML projects
"""

print(takeaways)

print("\n" + "=" * 70)
print("All visualization files saved!")
print("Next: Move to ML model building")
print("=" * 70)


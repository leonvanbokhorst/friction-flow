import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(42)  # Because we're responsible citizens 😉

# Let's create some correlated data - imagine these are butterfly positions!
n_samples = 300
theta = np.random.uniform(0, 2 * np.pi, n_samples)
r = 2 + np.random.normal(0, 0.1, n_samples)

# Create a squished circular pattern
X = np.column_stack(
    [
        r * np.cos(theta),  # x coordinate
        0.5 * r * np.sin(theta),  # y coordinate (squished)
    ]
)

# Add some random noise - butterflies are never perfectly aligned!
X += np.random.normal(0, 0.1, X.shape)

# Let's do the PCA magic!
pca = PCA()
X_pca = pca.fit_transform(X)

# Time for some beautiful visualizations!
plt.figure(figsize=(15, 5))

# Original data
plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title("Original Butterfly Cloud")
plt.axis("equal")

# Data with principal components
plt.subplot(132)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp_line = comp * var  # Scale by variance
    plt.arrow(
        0,
        0,
        comp_line[0],
        comp_line[1],
        color=f"C{i}",
        width=0.1,
        head_width=0.3,
        label=f"PC{i+1}",
    )
plt.title("Found the Main Flight Patterns!")
plt.axis("equal")
plt.legend()

# Explained variance
plt.subplot(133)
plt.bar(["PC1", "PC2"], pca.explained_variance_ratio_ * 100)
plt.title("How Important is Each Direction?")
plt.ylabel("Variance Explained (%)")

plt.tight_layout()
plt.show()

print("\n🦋 Fun Facts about our Butterfly Data:")
print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of the variation")
print(f"PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of the variation")

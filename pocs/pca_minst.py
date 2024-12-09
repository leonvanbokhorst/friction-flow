"""
Principal Component Analysis (PCA) Demonstration with MNIST Dataset

PCA Algorithm Overview:
1. Data Preparation:
   - Center the data by subtracting the mean
   - (Optional) Scale the features to have unit variance
   
2. Covariance Matrix:
   - Compute the covariance matrix of the centered data
   - This matrix captures how features vary together
   
3. Eigendecomposition:
   - Find eigenvectors and eigenvalues of the covariance matrix
   - Eigenvectors become the principal components (directions of max variance)
   - Eigenvalues indicate how much variance each component captures
   
4. Component Selection:
   - Sort eigenvectors by their eigenvalues (descending)
   - Select top k components for dimensionality reduction
   
5. Data Transformation:
   - Project original data onto selected principal components
   - This creates a lower-dimensional representation
   
6. Reconstruction (Optional):
   - Project reduced data back to original space
   - Results in an approximation of original data

Mathematical Foundation:
- If X is our data matrix (n_samples × n_features)
- Covariance matrix C = (1/n) * X^T * X
- Find eigenvectors V and eigenvalues λ where C*V = V*λ
- Transformed data = X * V[:, :k] (keeping k components)

Key Properties:
- Components are orthogonal (perpendicular to each other)
- First component captures maximum variance
- Each subsequent component captures maximum remaining variance
- Components are uncorrelated with each other
"""

from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load MNIST - but let's just take a subset to keep things snappy
print("🎲 Fetching some handwritten digits...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
# Each image is 28x28 = 784 pixels, flattened into a 1D array
# X shape is (2000, 784) after subsetting
X = X[:2000]  # Let's take 2000 images to start
y = y[:2000]

# Scale the data between 0 and 1
# This is important for PCA as it's sensitive to the scale of input features
# Without scaling, features with larger ranges could dominate the principal components
X = X / 255.0

# Display original digit
plt.figure(figsize=(4, 4))
plt.imshow(X[0].reshape(28, 28), cmap="gray")
plt.title("Original Digit")
plt.axis("off")
plt.show()

# PCA Compression Experiment
# We'll try different numbers of principal components to see the trade-off
# between compression ratio and image quality
n_components_list = [10, 30, 50, 100]
fig, axes = plt.subplots(1, len(n_components_list) + 1, figsize=(15, 3))

# Show original image (784 dimensions)
axes[0].imshow(X[0].reshape(28, 28), cmap="gray")
axes[0].set_title("Original\n(784 pixels)")
axes[0].axis("off")

# Initialize PCA with maximum number of components needed
max_components = max(n_components_list)
pca = PCA(n_components=max_components)

# Fit and transform the data once
X_transformed = pca.fit_transform(X)

# Try different compression levels by using subset of components
for idx, n_comp in enumerate(n_components_list, 1):
    # Slice the transformed data and components for reconstruction
    X_transformed_subset = X_transformed[:, :n_comp]
    components_subset = pca.components_[:n_comp]

    # Reconstruct using subset of components
    X_reconstructed = np.dot(X_transformed_subset, components_subset)
    # Add the mean back since PCA removes it during fitting
    X_reconstructed = X_reconstructed + pca.mean_

    # Display reconstructed image
    axes[idx].imshow(X_reconstructed[0].reshape(28, 28), cmap="gray")
    # Show compression ratio - how much of original data we're using
    axes[idx].set_title(f"{n_comp} components\n({n_comp/784*100:.1f}% of data)")
    axes[idx].axis("off")

    # explained_variance_ratio_ shows how much variance each principal component captures
    # The sum tells us total variance preserved in our compressed representation
    print(
        f"With {n_comp} components, we capture {pca.explained_variance_ratio_.sum()*100:.1f}% of the variance"
    )

plt.tight_layout()
plt.show()

# Let's look at the first 16 principal components - the "building blocks" of digits
pca = PCA(n_components=16)
pca.fit(X)

# Plot the eigendigits
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.ravel()):
    eigenvector = pca.components_[i].reshape(28, 28)
    ax.imshow(eigenvector, cmap="RdBu")
    ax.axis("off")
    ax.set_title(f"Component {i+1}")
plt.suptitle("The Building Blocks of Digits!", fontsize=16)
plt.tight_layout()
plt.show()

# Let's also show how much each component contributes
plt.figure(figsize=(10, 4))
plt.plot(pca.explained_variance_ratio_[:16] * 100, "bo-")
plt.title("Importance of Each Component")
plt.xlabel("Component Number")
plt.ylabel("Variance Explained (%)")
plt.grid(True)
plt.show()

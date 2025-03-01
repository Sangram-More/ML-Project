import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Generate a random 3D dataset with correlation
np.random.seed(42)
n_samples = 100
X = np.random.multivariate_normal([5, 5, 5], [[3, 2, 1], [2, 3, 1], [1, 1, 2]], n_samples)

# Apply PCA to reduce 3D data to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Get PCA components (eigenvectors)
pca_components = pca.components_
mean_X = np.mean(X, axis=0)

# Create a function to animate the 3D scatter plot
def animate_3D(i, scatter, vectors, ax):
    ax.view_init(elev=20, azim=i)
    return scatter, vectors

# Create figure for original 3D data
fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.add_subplot(111, projection='3d')
scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c='teal', s=20)
ax1.set_title("Original 3D Data")
ax1.set_xlabel("X1")
ax1.set_ylabel("X2")
ax1.set_zlabel("X3")

# Plot original data vectors
vectors1 = [ax1.quiver(mean_X[0], mean_X[1], mean_X[2], 
                       pca_components[0, 0], pca_components[0, 1], pca_components[0, 2], color='red', linewidth=2),
            ax1.quiver(mean_X[0], mean_X[1], mean_X[2], 
                       pca_components[1, 0], pca_components[1, 1], pca_components[1, 2], color='blue', linewidth=2)]

ani1 = animation.FuncAnimation(fig1, animate_3D, frames=360, interval=20, fargs=(scatter1, vectors1, ax1))

# Reconstruct the PCA components into 3D space for visualization
X_pca_3D = pca.inverse_transform(X_pca)
mean_X_pca = np.mean(X_pca_3D, axis=0)

# Create figure for transformed 3D data
fig2 = plt.figure(figsize=(6, 6))
ax2 = fig2.add_subplot(111, projection='3d')
scatter2 = ax2.scatter(X_pca_3D[:, 0], X_pca_3D[:, 1], X_pca_3D[:, 2], c='crimson', s=20)
ax2.set_title("PCA Transformed Data")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_zlabel("PC3")

# Plot PCA-transformed vectors
vectors2 = [ax2.quiver(mean_X_pca[0], mean_X_pca[1], mean_X_pca[2], 
                       pca_components[0, 0], pca_components[0, 1], pca_components[0, 2], color='red', linewidth=2),
            ax2.quiver(mean_X_pca[0], mean_X_pca[1], mean_X_pca[2], 
                       pca_components[1, 0], pca_components[1, 1], pca_components[1, 2], color='blue', linewidth=2)]

ani2 = animation.FuncAnimation(fig2, animate_3D, frames=360, interval=20, fargs=(scatter2, vectors2, ax2))

# # Save animation data as JSON
# animation_data = {
#     "original_data": X.tolist(),
#     "pca_transformed_data": X_pca_3D.tolist(),
#     "pca_components": pca_components.tolist()
# }

# with open("pca_animation_data.json", "w") as json_file:
#     json.dump(animation_data, json_file)

ani1.save("pca_original.gif", writer="pillow", fps=30)
ani2.save("pca_transformed.gif", writer="pillow", fps=30)


plt.show()

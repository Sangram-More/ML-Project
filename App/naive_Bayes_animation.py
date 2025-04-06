# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from sklearn.naive_bayes import GaussianNB
# from sklearn.datasets import make_classification

# # Generate synthetic 2D classification data
# X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
#                            n_redundant=0, n_clusters_per_class=1, random_state=42)

# # Sort by class label for animation effect
# indices = np.argsort(y)
# X, y = X[indices], y[indices]

# fig, ax = plt.subplots(figsize=(6, 5))

# clf = GaussianNB()
# step = 10
# max_frames = len(X) // step

# def animate(i):
#     ax.clear()
#     current_X = X[: (i + 1) * step]
#     current_y = y[: (i + 1) * step]

#     clf.fit(current_X, current_y)

#     # Plot decision boundary
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
#                          np.linspace(y_min, y_max, 100))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

#     ax.scatter(current_X[:, 0], current_X[:, 1], c=current_y, cmap='coolwarm', edgecolor='k')
#     ax.set_title(f"Frame {i+1}: Trained on {(i + 1) * step} samples")

# anim = FuncAnimation(fig, animate, frames=max_frames, interval=500)
# anim.save("gaussian_nb_animation.gif", writer='pillow')
# print("✅ Saved: gaussian_nb_animation.gif")

# --------------------------------------------------------------------------------------------------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.datasets import make_classification
# from sklearn.preprocessing import Binarizer

# # Generate binary classification data
# X, y = make_classification(n_samples=200, n_features=2, n_informative=2, 
#                            n_redundant=0, n_clusters_per_class=1, random_state=42)

# # Convert features to binary using thresholding
# binarizer = Binarizer(threshold=0.0)
# X_bin = binarizer.fit_transform(X)

# # Sort for smooth animation
# indices = np.argsort(y)
# X_bin, y = X_bin[indices], y[indices]

# # Create plot
# fig, ax = plt.subplots(figsize=(6, 5))
# clf = BernoulliNB()
# step = 10
# max_frames = len(X_bin) // step

# def animate(i):
#     ax.clear()
#     current_X = X_bin[: (i + 1) * step]
#     current_y = y[: (i + 1) * step]

#     clf.fit(current_X, current_y)

#     # Create meshgrid of binary features (0 or 1)
#     xx, yy = np.meshgrid([0, 1], [0, 1])
#     grid = np.c_[xx.ravel(), yy.ravel()]
#     Z = clf.predict(grid).reshape(xx.shape)

#     # Plot binary decision space
#     ax.pcolormesh(xx, yy, Z, cmap='coolwarm', alpha=0.3, shading='auto')
#     ax.scatter(current_X[:, 0], current_X[:, 1], c=current_y, cmap='coolwarm', edgecolor='k')
#     ax.set_xlim(-0.5, 1.5)
#     ax.set_ylim(-0.5, 1.5)
#     ax.set_xticks([0, 1])
#     ax.set_yticks([0, 1])
#     ax.set_title(f"BernoulliNB: Trained on {(i + 1) * step} samples")

# anim = FuncAnimation(fig, animate, frames=max_frames, interval=700)

# # Save animation
# anim.save("bernoulli_nb_animation.gif", writer='pillow')
# print("✅ Saved: bernoulli_nb_animation.gif")

# --------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

# 1. Generate synthetic continuous dataset
np.random.seed(42)
X = np.random.rand(200, 2) * 10  # 2 features
y = (X[:, 0] + X[:, 1] > 10).astype(int)  # class boundary

# 2. Discretize the features to create categorical inputs
kbins = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
X_cat = kbins.fit_transform(X).astype(int)

# 3. Initialize the model and parameters
clf = CategoricalNB()
fig, ax = plt.subplots(figsize=(6, 6))

# 4. Dynamically determine the number of bins for the animation grid
n_bins_0 = len(np.unique(X_cat[:, 0]))
n_bins_1 = len(np.unique(X_cat[:, 1]))
grid_vals = np.array([[i, j] for i in range(n_bins_0) for j in range(n_bins_1)])

# 5. Animation function
step = 10  # number of points added per frame

def animate(i):
    ax.clear()
    current_X = X_cat[: (i + 1) * step]
    current_y = y[: (i + 1) * step]

    clf.fit(current_X, current_y)
    Z = clf.predict(grid_vals).reshape(n_bins_0, n_bins_1)

    # Plot decision boundaries
    xx, yy = np.meshgrid(range(n_bins_0), range(n_bins_1))
    ax.pcolormesh(xx, yy, Z.T, cmap='coolwarm', alpha=0.3, shading='auto')

    # Plot training points
    ax.scatter(current_X[:, 0], current_X[:, 1], c=current_y, cmap='coolwarm', edgecolor='k')
    ax.set_title(f"Categorical Naive Bayes - Trained on {(i + 1) * step} samples")
    ax.set_xlabel("Feature 1 (Categorical)")
    ax.set_ylabel("Feature 2 (Categorical)")
    ax.set_xlim(-0.5, n_bins_0 - 0.5)
    ax.set_ylim(-0.5, n_bins_1 - 0.5)
    ax.set_xticks(range(n_bins_0))
    ax.set_yticks(range(n_bins_1))

# 6. Create animation
frames = len(X_cat) // step
anim = animation.FuncAnimation(fig, animate, frames=frames, interval=500)

# 7. Save as GIF
anim.save("categorical_nb_animation.gif", writer='pillow')
plt.show()

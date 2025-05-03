import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from preprocess import load_dataset
from sklearn.preprocessing import StandardScaler

# Class mappings (folder name to target class)
class_mapping = {
    # 'left_hand': 'left_hand',
    # 'left_hand_w_finger_movement': 'left_hand',
    # 'right_hand': 'right_hand',
    # 'right_hand_w_finger_movement': 'right_hand',
    # 'relaxed_state': 'resting',
    # 'meditative_state': 'resting',
    'both_hand': 'flexing',
    'left_hand': 'flexing',
    # 'left_hand_w_finger_movement': 'flexing',
    'right_hand': 'flexing',
    # 'right_hand_w_finger_movement': 'flexing',
    'relaxed_state': 'resting',
    'meditative_state': 'resting'
}

# Load model and data
model = joblib.load("eeg_svm_model.joblib")
X, y = load_dataset("eeg_data", class_mapping, normalize_len=True)  # reuse your function

X = StandardScaler().fit_transform(X)

# Encode class labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Reduce feature space to 2D with PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)

# Predict decision function over a grid
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

grid = np.c_[xx.ravel(), yy.ravel()]
grid_original_space = pca.inverse_transform(grid)
Z = model.decision_function(grid_original_space)
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.7)
plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)  # decision boundary
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='bwr', edgecolors='k')
plt.legend(handles=scatter.legend_elements()[0], labels=list(label_encoder.classes_))
plt.title("SVM Decision Boundary (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()



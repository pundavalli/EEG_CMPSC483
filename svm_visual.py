import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import numpy as np
from preprocess import load_dataset

# Class mappings (folder name to target class)
class_mapping = {
        # 'left_hand': 'left_hand',
        # 'left_hand_w_finger_movement': 'left_hand',
        # 'right_hand': 'right_hand',
        # 'right_hand_w_finger_movement': 'right_hand',
        'both_hand': 'flexing',
        'left_hand': 'flexing',
        # 'left_hand_w_finger_movement': 'flexing',
        'right_hand': 'flexing',
        # 'right_hand_w_finger_movement': 'flexing',
        #'both_hand': 'both_hand',
        'relaxed_state': 'resting',
        'meditative_state': 'resting',
        #'home_right': 'flexing',
        #'home_left': 'flexing',
        #'home_both': 'flexing',
        'home_rest': 'resting',
}

# X: your feature matrix
# y: your labels
X, y = load_dataset("eeg_data", class_mapping)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Reduce to 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Fit the model again on reduced data (needed for boundary)
from sklearn.svm import SVC
svm_2d = SVC(kernel='rbf', C=10)
svm_2d.fit(X_reduced, y_encoded)

# Create a meshgrid for plotting
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_encoded, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title("Non-linear SVM Decision Boundary (PCA-reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)

import matplotlib.patches as mpatches

# Create legend handles
classes = label_encoder.classes_
colors = [plt.cm.coolwarm(i / (len(classes)-1)) for i in range(len(classes))]
handles = [mpatches.Patch(color=colors[i], label=classes[i]) for i in range(len(classes))]

plt.legend(handles=handles)

plt.savefig("svm_decision_boundary.png")
plt.show()

'''
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_encoded, cmap=plt.cm.coolwarm)
plt.title("t-SNE Visualization of EEG Data")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.show()
'''
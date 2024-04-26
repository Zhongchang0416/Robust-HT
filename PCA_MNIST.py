import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset

# Load MNIST dataset
mnist = datasets.MNIST(root='./data', train=True, download=True)

# Select one image from the dataset
image_index = 0  # Change this index to select a different image
single_image = mnist.data[image_index]
print(single_image.shape)
# Preprocess the single image
single_image_scaled = StandardScaler().fit_transform(single_image)

# Apply PCA
pca = PCA(n_components=2)
single_image_pca = pca.fit_transform(single_image_scaled)

print(single_image_pca.shape)
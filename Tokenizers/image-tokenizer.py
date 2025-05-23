import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

root_dir = "C:\\Users\\shour\\OneDrive - vit.ac.in\\torch-implementations\\Malaria"
classes = os.listdir(root_dir)

image_paths = []

for img_class in classes:
    path = root_dir + "\\" + img_class
    images = os.listdir(path)
    image_paths.extend([path + "\\" + img for img in images])


class ImageTokenizer:
    def __init__(self, window_size : tuple[int, int]):
        self.window_size = window_size

    def tokenize(self, img):
        H, W, C = img.shape
        wh, ww = self.window_size
        expected_h = ((H + wh - 1) // wh) * wh
        expected_w = ((W + ww - 1) // ww) * ww
        difference = tuple(((exp - orig) // 2, (exp - orig) // 2) for exp, orig in zip((expected_h, expected_w, C), (H, W, C)))
        padded_array = np.pad(img, difference, mode='constant', constant_values=0)
        return padded_array.reshape((padded_array.shape[0] // wh) * (padded_array.shape[1] // ww), wh, ww, C)
        
tokenizer = ImageTokenizer((32, 32))

img_path = image_paths[0]

img = Image.open(img_path)
img_array = np.array(img) / 255

tokens = tokenizer.tokenize(img_array)
print(tokens.shape)
num_images = 36

fig, axes = plt.subplots(6, 6, figsize=(8, 8))

for i in range(num_images):
    row, col = i // 6, i % 6
    axes[row, col].imshow(tokens[i])  
    axes[row, col].axis('off')             
    axes[row, col].set_title(f"Image {i+1}")

plt.tight_layout()
plt.show()
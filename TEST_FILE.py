import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import tifffile
import torch.nn as nn
import torch



# # Test the train image saving
b = np.load('../data/Cell_seg_C_X_Y/train/Day_5_R1_WB_ROI1_all_markers_1533_3784.ome_244_244.ome.npz')

# print(list(b.keys()))

# print(b["label"].shape)

# print("\n\nTest:\n\n")
# # Test the test image saving
# hf = h5py.File('../data/Cell_seg/data_archive_8.npy.h5')
# data = hf.keys()

# print(data)

# print(hf.get("image").shape)
# print(hf.get("label").shape)


# Test the 2D Conv Method:

proj = nn.Conv2d(in_channels = 7, out_channels = 96, kernel_size=(4, 4), stride=(4, 4))


img = b["image"]
img = torch.from_numpy(img.astype(np.float32))
print("original image shape: ", img.shape)

conv_img = proj(img)
print("processed image shape: ", conv_img.shape)

print("\n________________________________\n")

proj = nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size=(4, 4), stride=(4, 4))

img = b["image"][:3]
img = torch.from_numpy(img.astype(np.float32))
print("original image shape: ", img.shape)

conv_img = proj(img)
print("processed image shape: ", conv_img.shape)
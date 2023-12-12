import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import tifffile



# Test the train image saving
b = np.load('../data/Cell_seg/Day_5_R1_WB_ROI1_all_markers_456_1917.npz')
print(list(b.keys()))

print(b["image"].shape)
print(b["label"].shape)

print("\n\nTest:\n\n")
# Test the test image saving
hf = h5py.File('../data/Cell_seg/data_archive_8.npy.h5')
data = hf.keys()

print(data)

print(hf.get("image").shape)
print(hf.get("label").shape)
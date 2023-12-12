import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import tifffile

import torch
import torch.nn as nn



def load_and_save_test_images(path_to_folder):
    """
    Load TIFF images and corresponding labels with '_cp_masks.png' suffix from the specified directory
    and save them in a numpy .npz file.
    
    Example usage:
    load_and_save_images("/path/to/input_directory", "/path/to/output_file.npy")

    Parameters:
    - input_dir (str or Path): 
        Path to the directory containing TIFF images and label files.
    - output_dir (str or Path): 
        Path to the output .npy file.

    Returns:
        None
    """
    path_to_folder = Path(path_to_folder)

    # Get a list of TIFF image files in the input directory
    tiff_files = sorted(path_to_folder.glob('*.tif'))  # Update the pattern as needed

    for tiff_file in tiff_files:
        # Load TIFF image using tifffile
        image = np.array(tifffile.imread(tiff_file))
        print(image.shape)

        # Construct the corresponding label file path with '_cp_masks.png' suffix
        label_file = path_to_folder / f"{tiff_file.stem}_cp_masks.png"

        # Check if the corresponding label file exists
        if label_file.exists():
            # Load label image using PIL
            label = np.array(Image.open(label_file))
            # Save images and labels in a numpy .npy file
            np.savez(path_to_folder / f"{tiff_file.stem}.npz", image = image, label = label)
        else:
            print("Label file could not be found")
            exit()



def load_and_save_train_images(path_to_folder):
    """
    Load three pairs of TIFF images and corresponding labels with '_cp_masks.png' suffix from the specified directory
    and save them in a single numpy .npy.h5 file.


    Example usage:
    load_and_save_images("/path/to/input_directory", "/path/to/output_file.npy.h5")

    Parameters:
    - path_to_folder (str or Path): 
        Path to the directory containing TIFF images and label files.

    Returns:
        None
    """
    path_to_folder = Path(path_to_folder)

    # Get a list of TIFF image files in the input directory
    tiff_files = sorted(path_to_folder.glob('*.tif'))  # Update the pattern as needed

    # Initialize empty lists for images and labels
    images = []
    labels = []


    for i, tiff_file in enumerate(tiff_files):
        # Load TIFF image using tifffile
        image = np.array(tifffile.imread(tiff_file))

        # Construct the corresponding label file path with '_cp_masks.png' suffix
        label_file = path_to_folder / f"{tiff_file.stem}_cp_masks.png"

        # Check if the corresponding label file exists
        if label_file.exists():
            # Load label image using PIL
            label = np.array(Image.open(label_file))

            # Append images and labels to the lists
            images.append(image)
            labels.append(label)

            # Check if three pairs of images and labels are loaded
            if len(images) == 3 and len(labels) == 3:
                # Convert lists to numpy arrays
                images = np.array(images)
                labels = np.array(labels)
                # Save images and labels in a single numpy .npy.h5 file
                with h5py.File(path_to_folder / f"data_archive_{i}.npy.h5", 'w') as hf:
                    hf.create_dataset('image', data=images)
                    hf.create_dataset('label', data=labels)

                # Clear the lists for the next batch of three pairs
                images = []
                labels = []
        else:
            print("Label file could not be found")
            exit()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
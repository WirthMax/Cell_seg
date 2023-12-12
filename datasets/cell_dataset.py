import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

from pathlib import Path


def random_rot_flip(image, label):
    """
    Apply random rotations (90-degree increments) and flips to the input image and label.

    Parameters:
    - image (numpy.ndarray): 
        Input image array.
    - label (numpy.ndarray): 
        Corresponding label array.

    Returns:
    - tuple (numpy.ndarray, numpy.ndarray): 
        A tuple containing the transformed image and label arrays after applying 
        random rotations and flips.
    """
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    """
    Apply a random rotation within the range of -20 to 20 degrees to the input image and label.

    Parameters:
    - image (numpy.ndarray): 
        Input image array.
    - label (numpy.ndarray): 
        Corresponding label array.

    Returns:
    - tuple (numpy.ndarray, numpy.ndarray): 
        A tuple containing the transformed image and label arrays after applying 
        a random rotation.
    """
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class Cell_dataset(Dataset):
    """
    Takes an index `idx` and returns the corresponding sample. 
    If split is "train", it loads data from an npz file, otherwise from an npy.h5 file. 
    The resulting 'image' and 'label' values are stored in a dictionary named sample.
    
    Parameters:
    - base_dir (str): 
        Base directory containing data files.
    - split (str): 
        Indicates if the dataset is for training ("train") or not.
    - transform (callable, optional): 
        Transformation function applied to the sample.
    """
    def __init__(self, base_dir, split, transform = None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = list((Path(base_dir) / f"{self.split}").glob("*.np*"))


    def __len__(self):
        """returns the length of the sample_list attribute, 
        which represents the number of samples in the dataset.

        Returns:
        - int: Number of samples in the dataset.
        """
        return len(self.sample_list)


    def __getitem__(self, idx):
        """
        Retrieves and returns the sample at the specified index.
        If transform is not None, the transform method is applied. 
        The 'case_name' key is assigned the sample name, and the dictionary is returned.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - sample (dict): 
            A dictionary with 'image', 'label', and 'case_name' keys 
            representing the sample data.
        """
        if self.split == "train":
            slice_name = self.sample_list[idx]
            data = np.load(slice_name)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx]
            data = h5py.File(vol_name)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].stem
        return sample


class RandomGenerator(object):
    """
    Randomly applies a combination of rotation, flipping, and resizing to an input image and label.

    Parameters:
    - output_size (tuple): 
        Desired output size for the image and label.
    """
    def __init__(self, output_size):

        self.output_size = output_size

    def __call__(self, sample):
        """
        Applies random transformations to the input image and label.

        Parameters:
        - sample (dict): 
            A dictionary containing 'image' and 'label' arrays.

        Returns:
        - sample (dict): 
            A dictionary with the transformed image and label arrays.
        """
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # get the dimensions in x, channels, y            
        x, c, y = image.shape
        # Resize the image and label to the desired output size using cubic interpolation (order=3)
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # Convert the image to a PyTorch tensor and add a channel dimension
        image = torch.from_numpy(image.astype(np.float32))#.unsqueeze(0)
        
        # Convert the label to a PyTorch tensor with long data type
        label = torch.from_numpy(label.astype(np.float32))
        
        # Create a dictionary with the transformed image and label
        sample = {'image': image, 'label': label.long()}




        return sample
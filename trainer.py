import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from Utils import DiceLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms


def cell_trainer(args, model, snapshot_path):
    """
    Train the SWIN-unet on cell dataset using the specified training arguments.

    Parameters:
    - args (argparse.Namespace): 
        Command-line arguments containing training configuration.
    - model (torch.nn.Module): 
        Neural network model to be trained.
    - snapshot_path (pathlib.Path): 
        Path to the directory for saving training snapshots and logs.

    Returns:
        None
    """
    from datasets.cell_dataset import Cell_dataset, RandomGenerator

    # Configure logging to save logs in a file and print to stdout
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Log the training configuration
    logging.info(str(args))

    # Extract training configuration parameters
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    max_iterations = args.max_iterations

    db_train = Cell_dataset(base_dir=args.root_path, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        """
        Worker initialization function for PyTorch DataLoader.

        This function sets the seed for each worker's random number generator based on the
        global seed and the worker's unique ID.

        Parameters:
        - worker_id (int): 
            The unique ID assigned to each worker by the DataLoader.

        Returns:
            None
        """
        random.seed(args.seed + worker_id)
    
    # initialize train loader
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, 
                                num_workers=8, pin_memory=True,
                                    worker_init_fn=worker_init_fn)

    # Apply data parallelism if using multiple GPUs
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    
    # Set the model to training mode and define loss functions
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, 
                            momentum=0.9, weight_decay=0.0001)

    # Create a TensorBoard summary writer for logging
    writer = SummaryWriter(snapshot_path + '/log')

    # Set up training loop parameters
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    # Initialize best performance metric for model checkpointing
    best_performance = 0.0

    # Set up tqdm iterator for visualizing training progress
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            break
        break


print("trainer.py done")

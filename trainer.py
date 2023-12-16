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
    
    def collate_fn(batch):
        return {
            'image': torch.stack([x['image'] for x in batch]),
            'label': torch.stack([x['label'] for x in batch])
        }

    # initialize train loader
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, 
                                num_workers=8, pin_memory=True,
                                #persistent_workers= True,
                                    worker_init_fn=worker_init_fn, 
                                        collate_fn=collate_fn)

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
            
            # Extract input (image) and target (label) batches and 
            # move batches to GPU for accelerated computation
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            # Forward pass: compute model predictions
            outputs = model(image_batch)
            # Calculate individual losses
            loss_ce = ce_loss(torch.squeeze(outputs), label_batch[:].float())
            loss_dice = dice_loss(outputs, label_batch[:].float(), softmax=True)

            # Combine losses using a weighted sum
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
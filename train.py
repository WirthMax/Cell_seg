import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import cell_trainer
from config import get_config

import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", 
                    help='path to config file', )
parser.add_argument('--root_path', type=str,
                    default='../data/Cell_seg', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Cell_seg', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Cell_seg', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
# if args.dataset == "Cell_seg":
#     args.root_path = os.path.join(args.root_path, "train_npz")
config = get_config(args)


if __name__ == "__main__":

    # Set up training environment and seed for reproducibility
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Define dataset configuration
    dataset_name = args.dataset
    dataset_config = {
        'Cell_seg': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Cell_seg',
            'num_classes': 1,
        },
    }

    # Adjust learning rate based on batch size
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    # Set dataset-specific configuration
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    # Create output directory if it does not exist
    output_dir_path = pathlib.Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Initialize and load the ViT segmentation model
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net.load_from(config)

    # Select the appropriate trainer based on the dataset
    trainer = {'Cell_seg': cell_trainer,}
    trainer[dataset_name](args, net, args.output_dir)

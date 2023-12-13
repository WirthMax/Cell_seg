
import torch
import torch.nn as nn
import numpy as np
import copy

from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        pass

    def load_from(self, config):
        """
        Model Loading Utility

        This method loads a pretrained model into the current model based on the 
        provided configuration.

        Parameters:
        - config: Configuration object containing model-related settings.

        Returns:
        None
        """
        # Extract the pretrained path from the configuration
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        
        # Check if a pretrained model path is provided
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            
            # Determine the device for loading the pretrained model 
            # (GPU if available, else CPU)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load the pretrained model dictionary
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            
            # Check if the loaded dictionary contains a 'model' key
            if "model"  not in pretrained_dict:
                print("---start load pretrained model by splitting---")

                # Remove prefix from keys and delete keys related to 'output'
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained model of swin encoder---")

            # Get the state dictionaries of the current model and the pretrained model
            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)

            # Update keys in the pretrained model's dictionary for swin encoder
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            
            # Filter out keys that do not match in shape between the pretrained and current model
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            # Load the pretrained model into the current model with strict=False
            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
 

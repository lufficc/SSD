  
# Implementing SSD 300--(Input image shape=[n,300,300]) where n is the number of channels

# We are following NVIDIA's approach where in
# --Using Resnet50 backbone
# --USing only first 4 residual layers of the Resnet ie dircrding conv5_x
#     and onward.
# -- All strides of conv4_x are made 1x1
# -- Addition of additional layers, same implementation as in the Resnet Paper

# Importing necessary modules
import torch
import torch.nn as nn  # for nn layer generation and combinations
# We will be using the resnet 50 model for our implementation
from torchvision.models.resnet import resnet50


class ResNet(nn.Module):
    # The resnet backbone is to be used as a feature provider.
    # For 300x300 we expect a 38x38 feature map

    def __init__(self):
        super().__init__()
        # Loading the full Resnet 50 backbone. This means that the we are currently only
        # importing the REsnet50s architecture. No weights or biases have been initialised.
        # WE will be using Xavier initialisation for that
        # Loading the full Resnet 50 backbone
        backbone = resnet50(pretrained=True)

        # Extracting the the required layers form the backbone
        # nn.sequential converts the individial components extracted from resnet in a list to
        # to continiuos nn object on which we can perform backprop
        # Lets call this as our feautre provider. This provied us with the very first feature map [38x38] with 1024 channels
        self.feature_provider = nn.Sequential(*list(backbone.children())[:7])

        # NOTE: But it is necessary to change the layer's stride else the feature provider will give a feature of 19x19
        # The conv4_x layer is the last object in our feature provider list.
        # Since stride arvariable in only the first block of a resnet layer we select the
        # last layer with the [-1] index and its first block with [0] in the self.feature_provider[-1][0]
        conv4_block1 = self.feature_provider[-1][0]

        conv4_block1.conv1.stride = (1, 1)  # changing the stride to 1x1
        conv4_block1.conv2.stride = (1, 1)  # changing the stride to 1x1
        conv4_block1.downsample[0].stride = (
            1, 1)  # changing the stride to 1x1

    def forward(self, x):
        # provides a feature map in a forward pass
        x = self.feature_provider(x)
        return x  # [1024,38,38]


class resnet50_SSD300(nn.Module):
    # Contains the full SSD300 model with additoinal layers and classification and localisation heads
    def __init__(self, cfg, backbone=ResNet()):
        super().__init__()

        self.feature_provider = backbone  # initialising our feature provider backbone
        self.label_num = cfg.MODEL.NUM_CLASSES  # number of COCO classes

        # contains all the feature maps's shapes as string for easy reference
        features_list = ["38x38", "19x19", "10x10", "5x5", "3x3", "1x1"]

        # a dictionary mapping having keys as the feature map shape as string
        feature_channel_dict = {"38x38": 1024,
                                "19x19": 512,
                                "10x10": 512,
                                "5x5": 256,
                                "3x3": 256,
                                "1x1": 256}

        # number of proposed priors per feauture in a feature map as given in the paper
        num_proir_box_dict = {"38x38": 4,
                              "19x19": 6,
                              "10x10": 6,
                              "5x5": 6,
                              "3x3": 4,
                              "1x1": 4}

        intermediate_channel_dict = {"19x19": 256,
                                     "10x10": 256,
                                     "5x5": 128,
                                     "3x3": 128,
                                     "1x1": 128}

        # intermediate channels for the additional layers
        self._make_additional_features_maps(
            features_list, feature_channel_dict, intermediate_channel_dict)

        self.loc = []
        self.conf = []

        # Generating localisatin heads and classification heads
        for feature_map_name in features_list:
            priors_boxes=num_proir_box_dict[feature_map_name]
            output_channel_from_feature_map=feature_channel_dict[feature_map_name]
            self.loc.append(nn.Conv2d(output_channel_from_feature_map, priors_boxes * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(output_channel_from_feature_map, priors_boxes * self.label_num, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _make_additional_features_maps(self, features_list, feature_channel_dict, intermediate_channel_dict):

        # input for additional layers come from one behinf it is coming from the
        input_list = features_list[:-1]
        # output is as stated for each additional layer
        output_list = features_list[1:]

        self.additional_blocks = []

        for i, (prev_feature_name, current_feature_name) in enumerate(zip(input_list, output_list)):
            if i < 3:  # for the first 3 additional features maps (19x19 , 10x10, 5x5) we use padding of 1 and stride of 2 for  
                       # the second convolution that generates additional map
                layer = nn.Sequential(
                    nn.Conv2d(feature_channel_dict[prev_feature_name],
                              intermediate_channel_dict[current_feature_name], kernel_size=1, bias=False),

                    
                    nn.BatchNorm2d(intermediate_channel_dict[current_feature_name]),  # an additional implementation by NVIDIA
                    nn.ReLU(inplace=True),

                    # the differentiating conv from other two (1x1 and 3x3) 
                    nn.Conv2d(intermediate_channel_dict[current_feature_name], feature_channel_dict[current_feature_name], kernel_size=3,
                              padding=1, stride=2, bias=False),
                    
                    nn.BatchNorm2d(feature_channel_dict[current_feature_name]),  # an additional implementation by NVIDIA
                    nn.ReLU(inplace=True),
                )
            else: # for the last additional features maps (3x3 and 1x1) we use padding of 0 and stride of 1 for  
                  # the second convolution that generates additional map
                layer = nn.Sequential(
                    nn.Conv2d(feature_channel_dict[prev_feature_name],
                              intermediate_channel_dict[current_feature_name], kernel_size=1, bias=False),
                    
                    nn.BatchNorm2d(intermediate_channel_dict[current_feature_name]),  # an additional implementation by NVIDIA
                    nn.ReLU(inplace=True),

                    # the differentiating conv from other three (19x19 , 10x10, 5x5)
                    nn.Conv2d(intermediate_channel_dict[current_feature_name], feature_channel_dict[current_feature_name],
                              kernel_size=3, bias=False),
                    
                    nn.BatchNorm2d(feature_channel_dict[current_feature_name]),  # an additional implementation by NVIDIA
                    nn.ReLU(inplace=True),
                )
            # adding the new feature map generator block to our arsenal    
            self.additional_blocks.append(layer)


        # converting into nn modules so that they can be added to pytorchs
        # computational graph and backprop can be performed
        self.additional_blocks = nn.ModuleList(self.additional_blocks)


    # Xavier initialising the weights
    def _init_weights(self):
        # making a list of all blocks in out SSD300 models
        # note that the backbone already has weights initialised so we are 
        # initialising only the newly created layer's weights
        layers = [*self.additional_blocks, *self.loc, *self.conf] # *list 
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)


    def forward(self, x):
        x = self.feature_provider(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        return detection_feed

@registry.BACKBONES.register('resnet50')
def resnet50(cfg, pretrained=True):
    model = resnet50_SSD300(cfg)
    # model_url = 'you_model_url'
    # if pretrained:
    #     model.init_from_pretrain(load_state_dict_from_url(model_url))
    return model
import torch
from torch import nn



def cnn_categorization_improved(netspec_opts):
    """
    Constructs a network for the base categorization model.

    Arguments
    --------
    netspec_opts: (dictionary), the network's architecture. It has the keys
                 'kernel_size', 'num_filters', 'stride', and 'layer_type'.
                 Each key holds a list containing the values for the
                corresponding parameter for each layer.
    Returns
    ------
     net: (nn.Sequential), the base categorization model
    """
    # instantiate an instance of nn.Sequential
    net = nn.Sequential()

    # add the hidden layers
    def _get_layer(params):
        layer_type = params["layer_type"]

        if layer_type == "conv":
            return nn.Conv2d(in_channels=params["in_channels"],
                             out_channels=params["out_channels"],
                             kernel_size=params["kernel"],
                             stride=params["stride"],
                             padding=params["padding"])
        elif layer_type == "bn":
            return nn.BatchNorm2d(num_features=params["num_features"])
        elif layer_type == "relu":
            return nn.ReLU()
        elif layer_type == "avg_pool":
            return nn.AvgPool2d(kernel_size=params["kernel"])
        elif layer_type == "max_pool":
            return nn.MaxPool2d(kernel_size=params["kernel"])
        elif layer_type == "dropout":
            return nn.Dropout(p=params["p"])
        elif layer_type == "linear":
            return nn.Linear(in_features=params["in_features"], out_features=params["out_features"])
        elif layer_type == "flatten":
            return nn.Flatten()
        else:
            return None


    for name, params in netspec_opts:
        net.add_module(
            name,
            _get_layer(params)
        )

    return net

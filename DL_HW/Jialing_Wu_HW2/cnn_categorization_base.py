from torch import nn


def cnn_categorization_base(netspec_opts):
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
    def _get_layer(layer_type, in_channels=0, out_channels=0, kernel=0, stride=0):
        if layer_type == "conv":
            pad = (kernel - 1) // 2
            return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel, stride=stride, padding=pad)
        elif layer_type == "bn":
            return nn.BatchNorm2d(num_features=in_channels)
        elif layer_type == "relu":
            return nn.ReLU()
        elif layer_type == "pool":
            return nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=0)
        else:
            return None

    L = len(netspec_opts["kernel"])

    prev_layer_output_channels = 3
    for i in range(L):
        kernel = netspec_opts["kernel"][i]
        out_channels = netspec_opts["num_filters"][i]
        in_channels = prev_layer_output_channels
        stride = netspec_opts["stride"][i]
        layer_type = netspec_opts["layer_type"][i]

        net.add_module(
            f"{layer_type}_{i // 3 + 1}" if i < L - 1 else "pred",
            _get_layer(
                layer_type,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                stride=stride
            ))

        if out_channels:
            prev_layer_output_channels = out_channels

    return net

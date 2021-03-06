from torch import nn
from sum_layer import Sum
from collections import OrderedDict

# import all other functions you may need


class SemanticSegmentationImproved(nn.Module):
    def __init__(self, netspec_opts):
        """

        Creates a fully convolutional neural network for the improve semantic segmentation model.


        Arguments
        ---------
        netspec_opts: (dictionary), the architecture of the base semantic network.

        """
        super(SemanticSegmentationImproved, self).__init__()
        
        # implement the improvement model architecture
        self.net = nn.ModuleDict()

        def _get_layer(params):
            layer_type = params["layer_type"]
            if layer_type == "conv":
                padding = (params["kernel_size"] - 1) // 2
                return nn.Conv2d(
                    in_channels=params["input_channels"],
                    out_channels=params["num_filters"],
                    kernel_size=params["kernel_size"],
                    stride=params["stride"],
                    padding=padding)
            elif layer_type == "bn":
                return nn.BatchNorm2d(num_features=params["input_channels"])
            elif layer_type == "relu":
                return nn.ReLU()
            elif layer_type == "convt":
                padding = (params["kernel_size"] - params["stride"]) // 2
                return nn.ConvTranspose2d(
                    in_channels=params["input_channels"],
                    out_channels=params["output_channels"],
                    kernel_size=params["kernel_size"],
                    stride=params["stride"],
                    groups=params["input_channels"],
                    padding=padding,
                    bias=False)
            elif layer_type == "skip":
                return nn.Conv2d(in_channels=params["input_channels"],
                                 out_channels=params["output_channels"],
                                 kernel_size=params["kernel_size"],
                                 stride=params["stride"],
                                 padding=0)
            elif layer_type == "sum":
                return Sum()
            elif layer_type == "max_pool":
                return nn.MaxPool2d(kernel_size=params["kernel_size"])
            elif layer_type == "avg_pool":
                return nn.AvgPool2d(kernel_size=params["kernel_size"])
            else:
                return None

        # add the hidden layers as specified in the handout
        # since the netspec_opts defined in the assignment is not hard to use, here we beautify it:
        netopt_dict = OrderedDict()
        L = len(netspec_opts["kernel_size"])
        for i in range(L):
            name = netspec_opts["name"][i]
            netopt_dict[name] = {
                "layer_type": netspec_opts["layer_type"][i],
                "kernel_size": netspec_opts["kernel_size"][i],
                "num_filters": netspec_opts["num_filters"][i],
                "stride": netspec_opts["stride"][i],
                "input": netspec_opts["input"][i],
            }
            # import pprint
            # pprint.pprint(netopt_dict)
            cur = netopt_dict[name]
            input_layer_name = cur["input"]
            if isinstance(input_layer_name, tuple):
                input_layer_name = input_layer_name[0]
            cur["input_channels"] = 3 if input_layer_name == "input" else netopt_dict[input_layer_name]["output_channels"]
            cur["output_channels"] = cur["num_filters"] if cur["num_filters"] > 0 else cur["input_channels"]

        for name in netopt_dict:
            self.net[name] = _get_layer(netopt_dict[name])

        self.netopt_dict = netopt_dict

    def forward(self, x):
        """
        Define the forward propagation of the improvement model.

        Arguments
        ---------
        x: (Tensor) of size (B x C X H X W) where B is the mini-batch size, C is the number of
            channels and H and W are the spatial dimensions. X is the input activation volume.

        Returns
        -------
        out: (Tensor) of size (B x C' X H x W), where C' is the number of classes.

        """

        # implement the forward propagation
        out_tensors = {}
        out_tensors['input'] = x
        for name, params in self.netopt_dict.items():
            input_layer_name = params["input"]
            if isinstance(input_layer_name, tuple):
                out_tensors[name] = self.net[name].forward(*([out_tensors[i] for i in input_layer_name]))
            else:
                out_tensors[name] = self.net[name].forward(out_tensors[input_layer_name])
            # print(output, out_tensors[output].shape)

        # return the final activation volume
        return out_tensors['pred']


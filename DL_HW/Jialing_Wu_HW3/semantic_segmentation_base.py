from torch import nn
from sum_layer import Sum
from collections import OrderedDict

class SemanticSegmentationBase(nn.Module):
    def __init__(self, netspec_opts):
        """

        Creates a fully convolutional neural network for the base semantic segmentation model. Given that there are
        several layers, we strongly recommend that you keep your layers in an nn.ModuleDict as described in
        the assignment handout. nn.ModuleDict mirrors the operations of Python dictionaries.

        You will specify the architecture of the module in the constructor. And then define the forward
        propagation in the forward method as described in the handout.

        Arguments
        ---------
        netspec_opts: (dictionary), the architecture of the base semantic network. netspec_opts has the keys
                                    1. kernel_size: (list) of size L where L is the number of layers
                                        representing the kernel sizes
                                    2. layer_type: (list) of size L indicating the type of each layer
                                    3. num_filters: (list) of size L representing the number of filters for each layer
                                    4. stride: (list) of size L indicating the striding factor of each layer
                                    5. input: (List) of size L containing the layer number of the inputs for each layer.

        """
        super(SemanticSegmentationBase, self).__init__()

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
            else:
                return None

        # add the hidden layers as specified in the handout
        # since the netspec_opts defined in the assignment is not hard to use, here we beautify it:
        netopt_dict = OrderedDict()
        layer_name_which_provides_output = OrderedDict()
        L = len(netspec_opts["kernel_size"])
        for i in range(L):
            name = netspec_opts["name"][i]
            netopt_dict[name] = {
                "layer_type": netspec_opts["layer_type"][i],
                "kernel_size": netspec_opts["kernel_size"][i],
                "num_filters": netspec_opts["num_filters"][i],
                "stride": netspec_opts["stride"][i],
                "input": netspec_opts["input"][i],
                "output": netspec_opts["output"][i],
            }
            cur = netopt_dict[name]
            layer_name_which_provides_output[cur["output"]] = name
            look_for_a_layer_provides_certain_output_for_channels = cur["input"] if isinstance(cur["input"], str) else cur["input"][0]
            cur["input_channels"] = 3 if cur["input"] == "input" else netopt_dict[layer_name_which_provides_output[look_for_a_layer_provides_certain_output_for_channels]]["output_channels"]
            cur["output_channels"] = cur["num_filters"] if cur["num_filters"] > 0 else cur["input_channels"]

        for name in netopt_dict:
            self.net[name] = _get_layer(netopt_dict[name])

        self.netopt_dict = netopt_dict

    def forward(self, x):
        """
        Define the forward propagation of the base semantic segmentation model here. Starting with the input, pass
        the output of each layer to the succeeding layer until the final layer. Return the output of final layer
        as the predictions.

        Arguments
        ---------
        x: (Tensor) of size (B x C X H X W) where B is the mini-batch size, C is the number of
            channels and H and W are the spatial dimensions. X is the input activation volume.

        Returns
        -------
        out: (Tensor) of size (B x C' X H x W) where C' is the number of classes.

        """

        # implement the forward propagation as defined in the handout
        out_tensors = {}
        out_tensors['input'] = x
        for name, params in self.netopt_dict.items():
            input = params["input"]
            output = params["output"]
            if isinstance(input, tuple):
                out_tensors[output] = self.net[name].forward(*([out_tensors[i] for i in input]))
            else:
                out_tensors[output] = self.net[name].forward(out_tensors[input])
            #print(output, out_tensors[output].shape)

        # return the final activation volume
        return out_tensors['pred']

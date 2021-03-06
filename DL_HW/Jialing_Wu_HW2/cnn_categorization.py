from create_dataset import create_dataset
from cnn_categorization_base import cnn_categorization_base
from cnn_categorization_improved import cnn_categorization_improved
from train import train
import torch
from torch import random, save
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, TensorDataset

# seed the random number generator. Remove the line below if you want to try different initializations
random.manual_seed(0)


def cnn_categorization(model_type="base",
                       data_path="image_categorization_dataset.pt",
                       contrast_normalization=False, whiten=False):
    """
    Invokes the dataset creation, the model construction and training functions

    Arguments
    --------
    model_type: (string), the type of model to train. Use 'base' for the base model and 'improved for the improved model. Default: base
    data_path: (string), the path to the dataset. This argument will be passed to the dataset creation function
    contrast_normalization: (boolean), specifies whether or not to do contrast normalization
    whiten: (boolean), specifies whether or not to whiten the data.

    """
    # Do not change the output path
    # but you can uncomment the exp_dir if you do not want to save the model checkpoints
    print(model_type, data_path, contrast_normalization, whiten)
    output_path = "{}_image_categorization_dataset.pt".format(model_type)
    exp_dir = "./{}_models".format(model_type)

    # train_ds is TensorDataset(training_tensor_data, training_labels)
    # val_ds is TensorDataset(validation_tensor_data, validation_labels)
    train_ds, val_ds = create_dataset(data_path, output_path, contrast_normalization, whiten)

    # specify the network architecture and the training policy of the models under
    # the respective blocks
    if model_type == "base":
        # create netspec_opts
        netspec_opts = {
            "kernel": [3, 0, 0, 3, 0, 0, 3, 0, 0, 8, 1],
            "num_filters": [16, 16, 0, 32, 32, 0, 64, 64, 0, 0, 16],
            "stride": [1, 0, 0, 2, 0, 0, 2, 0, 0, 1, 1],
            "layer_type": ["conv", "bn", "relu", "conv", "bn", "relu", "conv", "bn", "relu", "pool", "conv"]
        }


        # create train_opts
        train_opts = {
            "num_epochs": 25,
            "lr": 0.1,
            "momentum": 0.9,
            "batch_size": 128,
            "weight_decay": 0.0001,
            "step_size": 20,
            "gamma": 0.1
        }

        # create model base on tetspect_opts
        model = cnn_categorization_base(netspec_opts)


    elif model_type == "improved":
        # create netspec_opts
        netspec_opts = [
            # ("bn0", {"layer_type": "bn", "num_features": 3}),
            ("conv1", {"layer_type": "conv", "kernel": 3, "in_channels": 3, "out_channels":16, "stride": 1, "padding": 1}),
            ("bn1", {"layer_type": "bn", "num_features": 16}),
            ("relu1", {"layer_type": "relu"}),
            ("conv2", {"layer_type": "conv", "kernel": 3, "in_channels": 16, "out_channels":16, "stride": 1, "padding": 1}),
            ("bn2", {"layer_type": "bn", "num_features": 16}),
            ("relu2", {"layer_type": "relu"}),
            ("conv3", {"layer_type": "conv", "kernel": 3, "in_channels": 16, "out_channels": 16, "stride": 1, "padding": 1}),
            ("bn3", {"layer_type": "bn", "num_features": 16}),
            ("relu3", {"layer_type": "relu"}),
            ("conv4", {"layer_type": "conv", "kernel": 3, "in_channels": 16, "out_channels": 32, "stride": 2, "padding": 1}),
            ("bn4", {"layer_type": "bn", "num_features": 32}),
            ("relu4", {"layer_type": "relu"}),
            ("conv5", {"layer_type": "conv", "kernel": 3, "in_channels": 32, "out_channels": 32, "stride": 1, "padding": 1}),
            ("bn5", {"layer_type": "bn", "num_features": 32}),
            ("relu5", {"layer_type": "relu"}),
            ("conv6", {"layer_type": "conv", "kernel": 3, "in_channels": 32, "out_channels": 64, "stride": 2, "padding": 1}),
            ("bn6", {"layer_type": "bn", "num_features": 64}),
            ("relu6", {"layer_type": "relu"}),
            ("conv7", {"layer_type": "conv", "kernel": 3, "in_channels": 64, "out_channels": 128, "stride": 1, "padding": 1}),
            ("bn7", {"layer_type": "bn", "num_features": 128}),
            ("relu7", {"layer_type": "relu"}),
            ("max_pool1", {"layer_type": "max_pool", "kernel": 8}),
            ("flatte1", {"layer_type": "flatten"}),
            ("linear1", {"layer_type": "linear", "in_features": 128, "out_features": 64}),
            ("relu8", {"layer_type": "relu"}),
            ("dropout1", {"layer_type": "dropout", "p": 0.3}),
            ("linear2", {"layer_type": "linear", "in_features": 64, "out_features": 64}),
            ("relu9", {"layer_type": "relu"}),
            ("dropout2", {"layer_type": "dropout", "p": 0.1}),
            ("pred", {"layer_type": "linear", "in_features": 64, "out_features": 16})
        ]

        # create train_opts
        train_opts = {
            "num_epochs": 180,
            "lr": 0.1,
            "momentum": 0.9,
            "batch_size": 256,
            "weight_decay": 0.0001,
            "step_size": 165,
            "gamma": 0.1
        }

        # create improved model
        model = cnn_categorization_improved(netspec_opts)

        #data augment
        class AugmentedDataset(Dataset):
            """TensorDataset with support of transforms.
            """
            def __init__(self, X, y, transforms=None):

                assert X.size(0) == y.size(0)
                self.X = X
                self.y = y
                self.tensors = (X, y)
                self.transforms = transforms

            def __getitem__(self, index):
                item_x = self.X[index]
                if self.transforms:
                    for transform in self.transforms:
                        item_x = transform(item_x)
                item_y = self.y[index]
                return item_x, item_y

            def __len__(self):
                '''
                Returns number of samples/items
                -------
                '''
                return self.X.size(0)


        # augment_functions
        def random_h_flip(x):
            """Flips tensor horizontally.
            """
            if torch.rand(1)[0].numpy() > 0.5:
                x = x.flip(2)
            return x

        def noise(x):
            """Add Gaussian Noise.
            """
            return x + torch.randn(x.size()) * 2

        def pad_and_crop(x):
            """pad 2 pixels and then crop to the original size
            """
            n_channel, n_h, n_w = x.shape
            target = torch.zeros(n_channel, n_h + 4, n_w + 4)
            target[:, 2:n_h+2, 2:n_w+2] = x
            i = int(torch.rand(1).tolist()[0] * 4)
            j = int(torch.rand(1).tolist()[0] * 4)
            x = target[:, i:i+n_h, j:j+n_w]
            return x


        def erase(x):
            """randomly erase a 2*2 area
            """
            n_channel, n_h, n_w = x.shape
            i = int(torch.rand(1).tolist()[0] * n_h)
            j = int(torch.rand(1).tolist()[0] * n_w)
            if i < n_h - 5 and j < n_w - 5:
                x[:, i:i+5, j:j+5] = 0
            return x


        # augment it!
        # train_ds is a TensorDataset, it has a field named 'tensors', in that, tensors[0] is the X and tensors[0] is the y
        origin_train_X, origin_train_y = train_ds.tensors
        train_ds = AugmentedDataset(X=origin_train_X, y=origin_train_y,
                                    transforms=[random_h_flip,
                                                pad_and_crop,
                                                noise,
                                                erase,
                                                erase])
    else:
        raise ValueError(f"Error: unknown model type {model_type}")

    # uncomment the line below if you wish to resume training of a saved model
    # model.load_state_dict(load(PATH to state))

    # train the model
    train(model, train_ds, val_ds, train_opts, exp_dir)

    # save model's state and architecture to the base directory
    state_dictionary_path = f"{model_type}_state_dict.pt"
    save(model.state_dict(), state_dictionary_path)
    model = {"state":state_dictionary_path, "specs": netspec_opts}
    save(model, "{}-model.pt".format(model_type))

    plt.savefig(f"{model_type}-categorization.png")
    plt.show()


if __name__ == '__main__':
    # Change the default values for the various parameters to your preferred values
    # Alternatively, you can specify different values from the command line
    # For example, to change model type from base to improved
    # type <cnn_categorization.py --model_type improved> at a command line and press enter
    args = ArgumentParser()
    args.add_argument("--model_type", type=str, default="base", required=False,
                      help="The model type must be either base or improved")
    args.add_argument("--data_path", type=str, default="image_categorization_dataset.pt",
                      required=False, help="Specify the path to the dataset")
    args.add_argument("--contrast_normalization", type=bool, default=False, required=False,
                      help="Specify whether or not to do contrast_normalization")
    args.add_argument("--whiten", type=bool, default=False, required=False,
                      help="Specify whether or not to whiten value")

    args, _ = args.parse_known_args()
    cnn_categorization(**args.__dict__)

from semantic_segmentation_base import SemanticSegmentationBase
from semantic_segmentation_improved import SemanticSegmentationImproved
from create_dataset import create_dataset
from train import train
from utils import distrib
import torch
from torch import optim
from torch import save, random
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from argparse import ArgumentParser
import matplotlib.pyplot as plt

# seeding the random number generator. You can disable the seeding for the improvement model
random.manual_seed(0)


def semantic_segmentation(model_type="base"):
    """
    sets up and trains a semantic segmentation model

    Arguments
    ---------
    model_type:  (String) a string in {'base', 'improved'} specifying the targeted model type
    """
    
    # the dataset
    train_dl, val_dl = create_dataset("semantic_segmentation_dataset.pt")

    # an optional export directory
    exp_dir = f"{model_type}_models"

    if model_type == "base":
        # specify netspec_opts
        netspec_opts = {
            "kernel_size": [3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 4, 1, 0, 4],
            "num_filters": [16, 16, 0, 32, 32, 0, 64, 64, 0, 128, 128, 0, 36, 36, 36, 0, 36],
            "stride": [1, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 1, 4, 1, 0, 2],
            "layer_type": ["conv", "bn", "relu", "conv", "bn", "relu", "conv", "bn", "relu",
                           "conv", "bn", "relu", "conv", "convt", "skip", "sum", "convt"],
            "input":["input", "conv_1", "bn_1", "relu_1", "conv_2", "bn_2", "relu_2", "conv3", "bn_3", "relu_3",
                       "conv_4", "bn_4", "relu_4", "conv_5", "relu_2", ("skip_6", "upsample_4x"), "betterFeat"],
            ###additional
            "name": ["conv_1", "bn_1", "relu_1", "conv_2", "bn_2", "relu_2", "conv3", "bn_3", "relu_3",
                     "conv_4", "bn_4", "relu_4", "conv_5", "upsample_4x", "skip_6", "sum_6", "upsample_2x"],
            "output": ["conv_1", "bn_1", "relu_1", "conv_2", "bn_2", "relu_2", "conv3", "bn_3", "relu_3",
                       "conv_4", "bn_4", "relu_4", "conv_5", "upsample_4x", "skip_6", "betterFeat", "pred"]
        }
        # specify train_opt
        train_opts = {
            "optimizer": "SGD",
            "num_epochs": 34,
            "lr": 0.1,
            "momentum": 0.9,
            "batch_size": 24,
            "weight_decay": 0.001,
            "step_size": 30,
            "gamma": 0.1,
            "objective": CrossEntropyLoss()
        }
        model = SemanticSegmentationBase(netspec_opts)

    elif model_type == "improved":
        class_count, _ = distrib(train_dl)
        print(class_count)
        class_count_sum = class_count.sum().float()
        class_weigths = class_count_sum / class_count
        class_weigths_sum = class_weigths.sum()
        class_weigths = class_weigths / class_weigths_sum
        print(class_weigths.float())
        # specify netspec_opts
        netspec_opts = {
            "kernel_size": [0, 3, 0, 0,
                            3, 0, 0,
                            3, 0, 0, 2,
                            3, 0, 0,
                            3, 0, 0, 2,
                            3, 0, 0,
                            3, 0, 0, 2,
                            3, 0, 0,
                            3, 0, 0, 2,
                            3, 0, 2,
                            2, 1, 0,
                            2, 1, 0,
                            2, 1, 0,
                            2, 1, 0,
                            2, 1, 0,
                            1],
            "num_filters": [0, 128, 128, 0,
                            128, 128, 0,
                            128, 128, 0, 0,
                            256, 256, 0,
                            256, 256, 0, 0,
                            512, 512, 0,
                            512, 512, 0, 0,
                            1024, 1024, 0,
                            1024, 1024, 0, 0,
                            36, 0, 0,
                            36, 36, 0,
                            36, 36, 0,
                            36, 36, 0,
                            36, 36, 0,
                            36, 36, 0,
                            36],
            # "num_filters": [0, 64, 64, 0,
            #                 64, 64, 0,
            #                 64, 64, 0, 0,
            #                 128, 128, 0,
            #                 128, 128, 0, 0,
            #                 256, 256, 0,
            #                 256, 256, 0, 0,
            #                 512, 512, 0,
            #                 512, 512, 0, 0,
            #                 36, 0, 0,
            #                 36, 36, 0,
            #                 36, 36, 0,
            #                 36, 36, 0,
            #                 36, 36, 0,
            #                 36, 36, 0,
            #                 36],
            "stride": [0, 1, 0, 0,
                       1, 0, 0,
                       1, 0, 0, 0,
                       1, 0, 0,
                       1, 0, 0, 0,
                       1, 0, 0,
                       1, 0, 0, 0,
                       1, 0, 0,
                       1, 0, 0, 0,
                       1, 0, 0,
                       2, 1, 0,
                       2, 1, 0,
                       2, 1, 0,
                       2, 1, 0,
                       2, 1, 0,
                       1],
            "layer_type": ["bn", "conv", "bn", "relu",
                           "conv", "bn", "relu",
                           "conv", "bn", "relu", "max_pool",
                           "conv", "bn", "relu",
                           "conv", "bn", "relu", "max_pool",
                           "conv", "bn", "relu",
                           "conv", "bn", "relu", "max_pool",
                           "conv", "bn", "relu",
                           "conv", "bn", "relu", "max_pool",
                           "conv", "relu", "max_pool",
                           "convt", "skip", "sum",
                           "convt", "skip", "sum",
                           "convt", "skip", "sum",
                           "convt", "skip", "sum",
                           "convt", "skip", "sum",
                           "conv"
                           ],
            "input": ["input", "bn_0", "conv_1_1", "bn_1_1",
                      "relu_1_1", "conv_1_2", "bn_1_2",
                      "relu_1_2", "conv_1_3", "bn_1_3", "relu_1_3",
                      "pool_1", "conv_2_1", "bn_2_1",
                      "relu_2_1", "conv_2_2", "bn_2_2", "relu_2_2",
                      "pool_2", "conv_3_1", "bn_3_1",
                      "relu_3_1", "conv_3_2", "bn_3_2", "relu_3_2",
                      "pool_3", "conv_4_1", "bn_4_1",
                      "relu_4_1", "conv_4_2", "bn_4_2", "relu_4_2",
                      "pool_4", "conv_5", "relu5",
                      "pool_5", "pool_4", ("skip_6", "convt_6_up_2x"),
                      "sum_6", "pool_3", ("skip_7", "convt_7_up_2x"),
                      "sum_7", "pool_2", ("skip_8", "convt_8_up_2x"),
                      "sum_8", "pool_1", ("skip_9", "convt_9_up_2x"),
                      "sum_9", "relu_1_3", ("skip_10", "convt_10_up_2x"),
                      "sum_10"
                      ],
            ###additional
            "name": ["bn_0", "conv_1_1", "bn_1_1", "relu_1_1",  #32
                     "conv_1_2", "bn_1_2", "relu_1_2",
                     "conv_1_3", "bn_1_3", "relu_1_3", "pool_1" ,  # 16
                     "conv_2_1", "bn_2_1", "relu_2_1",
                     "conv_2_2", "bn_2_2", "relu_2_2", "pool_2",  #8
                     "conv_3_1", "bn_3_1", "relu_3_1",
                     "conv_3_2", "bn_3_2", "relu_3_2","pool_3",  #4
                     "conv_4_1", "bn_4_1", "relu_4_1",
                     "conv_4_2", "bn_4_2", "relu_4_2","pool_4",  #2
                     "conv_5",  "relu5", "pool_5",  #1 but 36 channels(corresponding to 36 classes)
                     "convt_6_up_2x", "skip_6", "sum_6",  #2
                     "convt_7_up_2x", "skip_7", "sum_7",  #4
                     "convt_8_up_2x", "skip_8", "sum_8",  #8
                     "convt_9_up_2x", "skip_9", "sum_9",  #16
                     "convt_10_up_2x", "skip_10", "sum_10",  # 32
                     "pred"]
        }
        # specify train_opt
        train_opts = {
            "optimizer": "Adam",
            "num_epochs": 30,
            "lr": 0.001,
            "momentum": 0.9,
            "batch_size": 24,
            "weight_decay": 0.0001,
            "step_size": 27,
            "gamma": 0.1,
            "objective": CrossEntropyLoss(weight=class_weigths)
        }
        model = SemanticSegmentationImproved(netspec_opts)

        # data augment
        class AugmentedDataset(Dataset):
            """TensorDataset with support of transforms.
            """

            def __init__(self, X, Y, transforms=None):
                assert X.size(0) == Y.size(0)
                self.X = X
                self.Y = Y
                self.tensors = (X, Y)
                self.transforms = transforms

            def __getitem__(self, index):
                item_x = self.X[index]
                item_y = self.Y[index]
                if self.transforms:
                    for transform in self.transforms:
                        item_x, item_y = transform(item_x, item_y)
                return item_x, item_y.long()

            def __len__(self):
                '''
                Returns number of samples/items
                -------
                '''
                return self.X.size(0)

        # augment_functions
        def random_h_flip(x, y):
            """Flips tensor horizontally.
            """
            if torch.rand(1)[0].numpy() > 0.5:
                x = x.flip(2)
                y = y.flip(1)
            return x, y

        def noise(x, y):
            """Add Gaussian Noise.
            """
            return x + torch.randn(x.size()) * 2, y

        def rgb(x, y):
            """Add random brightness.
            """
            a = torch.rand(1)[0].numpy()
            b = torch.rand(1)[0].numpy()
            if b > 0.5:
                b = 1
            else:
                b= -1
            if a < 0.33:
                x[0, :, :] += int(torch.rand(1).tolist()[0]) * 2 * b
            elif a < 0.66:
                x[1, :, :] += int(torch.rand(1).tolist()[0]) * 2 * b
            else:
                x[2, :, :] += int(torch.rand(1).tolist()[0]) * 2 * b
            return x, y

        def pad_and_crop(x, y):
            """pad 2 pixels and then crop to the original size
            """
            n_channel, n_h, n_w = x.shape
            target_x = torch.zeros(n_channel, n_h + 4, n_w + 4)
            target_y = torch.zeros(n_h + 4, n_w + 4)
            target_x[:, 2:n_h + 2, 2:n_w + 2] = x
            target_y[2:n_h + 2, 2:n_w + 2] = y
            i = int(torch.rand(1).tolist()[0] * 4)
            j = int(torch.rand(1).tolist()[0] * 4)
            x = target_x[:, i:i + n_h, j:j + n_w]
            y = target_y[i:i + n_h, j:j + n_w]
            return x, y

        def erase(x, y):
            """randomly erase 16 1*1 area
            """
            n_channel, n_h, n_w = x.shape
            for _ in range(16):
                i = int(torch.rand(1).tolist()[0] * n_h)
                j = int(torch.rand(1).tolist()[0] * n_w)
                if i < n_h and j < n_w:
                    x[:, i, j] = 0
            return x, y

        # augment it!
        # train_ds is a TensorDataset, it has a field named 'tensors', in that, tensors[0] is the X and tensors[0] is the y
        origin_train_X, origin_train_Y = train_dl.tensors
        train_dl = AugmentedDataset(X=origin_train_X, Y=origin_train_Y,
                                    transforms=[random_h_flip,
                                                pad_and_crop,
                                                noise,
                                                rgb,
                                                erase])

    else:
        raise ValueError(f"Error: unknown model type {model_type}")


    # train the model
    train(model, train_dl, val_dl, train_opts, exp_dir=exp_dir)

    # save model's state and architecture to the base directory
    model = {"state": model.state_dict(), "specs": netspec_opts}
    save(model, f"{model_type}_semantic-model.pt")

    plt.savefig(f"{model_type}_semantic.png")
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_type", default="base", type=str, help="Specify model type")
    args, _ = parser.parse_known_args()

    semantic_segmentation(**args.__dict__)

from torch.utils.data import TensorDataset
import torch

def create_dataset(data_path):
    """
    Reads the data and prepares the training and validation sets. No preprocessing is required.

    Arguments
    ---------
    data_path: (string),  the path to the file containing the data

    Return
    ------
    train_ds: (TensorDataset), the examples (inputs and labels) in the training set
    val_ds: (TensorDataset), the examples (inputs and labels) in the validation set
    """

    dataset = torch.load(data_path)
    images_tr = dataset["images_tr"]  # this contains both train and validation: num 32000 + 6400 = 38400
    # images_mean = images_tr.mean(dim=0)
    # images_tr = images_tr - images_mean

    sets_tr = dataset["sets_tr"]
    anno_tr = dataset["anno_tr"]

    train_ds = TensorDataset(images_tr[sets_tr == 1], anno_tr[sets_tr == 1])
    val_ds = TensorDataset(images_tr[sets_tr == 2], anno_tr[sets_tr == 2])

    return train_ds, val_ds


if __name__ == '__main__':
    create_dataset("semantic_segmentation_dataset.pt")
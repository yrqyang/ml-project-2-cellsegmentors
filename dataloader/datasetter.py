from torch.utils.data import DataLoader
from monai.data import Dataset
import pickle

from .transforms import *
from .utils import split_train_valid, path_decoder

DATA_LABEL_DICT_PICKLE_FILE = ".dataloader/custom/modalities.pkl"

__all__ = ["get_dataloaders_labeled"]


def get_dataloaders_labeled(
    root,
    mapping_file,
    valid_portion=0.0,
    batch_size=1,
    amplified=False,
):
    """Set DataLoaders for labeled datasets.

    Args:
        root (str): root directory
        mapping_file (str): json file for mapping dataset
        valid_portion (float, optional): portion of valid datasets. Defaults to 0.1.
        batch_size (int, optional): batch size. Defaults to 8.
        shuffle (bool, optional): shuffles dataloader. Defaults to True.
        num_workers (int, optional): number of workers for each datalaoder. Defaults to 5.

    Returns:
        dict: dictionary of data loaders.
    """

    # Get list of data dictionaries from decoded paths
    data_dicts = path_decoder(root, mapping_file)

    if amplified:
        with open(DATA_LABEL_DICT_PICKLE_FILE, "rb") as f:
            data_label_dict = pickle.load(f)

        data_point_dict = {}

        for label, data_lst in data_label_dict.items():
            data_point_dict[label] = []

            for d_idx in data_lst:
                try:
                    data_point_dict[label].append(data_dicts[d_idx])
                except:
                    print(label, d_idx)

        data_dicts = []

        for label, data_points in data_point_dict.items():
            len_data_points = len(data_points)

            if len_data_points >= 50:
                data_dicts += data_points
            else:
                for i in range(50):
                    data_dicts.append(data_points[i % len_data_points])

    # Split datasets as Train/Valid
    train_dicts, valid_dicts = split_train_valid(
        data_dicts, valid_portion=valid_portion
    )

    # Obtain datasets with transforms
    trainset = Dataset(train_dicts, transform=train_transforms)
    validset = Dataset(valid_dicts, transform=valid_transforms)
    # tuningset = Dataset(tuning_dicts, transform=tuning_transforms)

    # Set dataloader for Trainset
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=5
    )

    # Set dataloader for Validset (Batch size is fixed as 1)
    valid_loader = DataLoader(validset, batch_size=1, shuffle=False,)

    # Form dataloaders as dictionary
    dataloaders = {
        "train": train_loader,
        "valid": valid_loader,
    }

    return dataloaders


from Imbalanced.imbalanced_methods import *
from dataset.dataset_methods import *


if __name__ == '__main__':
    # get dataset
    dataset = get_dataset()
    # dataset = dataset[:10]
    # split into X and y
    X, y = features_and_target(dataset)


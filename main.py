from imbalance_methods import *
from dataset import *

if __name__ == '__main__':
    # get dataset
    dataset = get_dataset()
    # dataset = dataset[:10]
    # split into X and y
    X, y = features_and_target(dataset)
    # New features -> tf-idf
    #

    # drop string features
    #

    # Oversampling
    # X_oversampling, y_oversampling = oversampling(X, y)
    # split into train and test sets
    # X_train, X_test, y_train, y_test = split_test(X_oversampling, y_oversampling)

    # Undersampling
    X_undersampling, y_undersampling = undersampling(X, y)
    # split into train and test sets
    X_train, X_test, y_train, y_test = split_test(X_undersampling, y_undersampling)



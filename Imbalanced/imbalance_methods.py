import imblearn


def oversampling(X, y):
    """
    Oversampling method -> SMOTE
    :param X:
    :param y:
    :return:
    """
    # SMOTE()
    X_oversampling, y_oversampling = imblearn.over_sampling.SMOTE().fit_resample(X, y)

    # returns X and y after SMOTE()
    return X_oversampling, y_oversampling


def undersampling(X, y):
    """
    Undersampling method -> TomekLinks
    :param X:
    :param y:
    :return:
    """
    # TomekLinks()
    X_undersampling, y_undersampling = imblearn.under_sampling.TomekLinks().fit_resample(X, y)

    # returns X and y after TomekLinks()
    return X_undersampling, y_undersampling

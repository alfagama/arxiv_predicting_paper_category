import pandas as pd
from sklearn.model_selection import train_test_split

# Options for pandas -----
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_preprocessed_dataset():
    """
    read preprocessed dataset from directory:
     "../data/preprocessed_dataset.csv"
    :return: dataset (dataframe)
    """
    # reads data/preprocessed_dataset.csv
    df = pd.read_csv("../data/preprocessed_conc_dataset.csv",
                     sep=',',
                     header=0,
                     skiprows=0)

    # print head
    print(df.head(5))

    # return dataframe
    return df


def get_dataset():
    """
    read preprocessed dataset from directory:
     "../data/preprocessed_dataset.csv"
    :return: dataset (dataframe)
    """
    # reads data/preprocessed_dataset.csv
    df = pd.read_csv("../data/preprocessed_dataset.csv",
                     sep=',',
                     header=0,
                     skiprows=0)

    # print head
    print(df.head(5))

    # return dataframe
    return df


def features_and_target(df):
    """
    Splits dataset in features and target column
    :param df: dataset (dataframe)
    :return: X -> features y -> target column
    """
    # splpit dataset in features and target column. Target -> 'categories'
    X = df.iloc[:, df.columns != "categories"].values
    y = df.iloc[:, df.columns == "categories"].values

    # return X and y
    return X, y


def split_xy_train_test(X, y):
    """
    Splits X and y into train and test set
    :param X: X -> features (dataframe columns)
    :param y: y -> target column (dataframe column)
    :return: X_train, X_test, y_train, y_test
    """
    # split X and y into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # return train and test sets
    return X_train, X_test, y_train, y_test


def split_df_train_test(df):
    """
    Splits df into train and test set
    :param df: dataframe
    :return: train, test
    """
    # split X and y into train and test set
    train, test = train_test_split(df, test_size=0.3, random_state=1)

    # return train and test sets
    return train, test

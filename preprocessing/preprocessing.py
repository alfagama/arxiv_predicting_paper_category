import pandas as pd

# Options for pandas -----
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def preprocess():
    """

    :return:
    """
    # reads data/ArXiv_dataset.csv
    df = pd.read_csv("../data/dataset.csv",
                     sep=',',
                     header=0,
                     skiprows=0)
    # print(data_set.head(5))

    # lower-case
    df.loc[:, "abstract"] = df.abstract.apply(lambda x: str.lower(x))
    df.loc[:, "categories"] = df.categories.apply(lambda x: str.lower(x))
    df.loc[:, "title"] = df.title.apply(lambda x: str.lower(x))
    print(df.head(5))

    #

    df.to_csv('data/preprocessed_dataset.csv')


preprocess()

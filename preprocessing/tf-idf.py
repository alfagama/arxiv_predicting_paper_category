from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt

# Options for pandas -----
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def tf_idf():
    """
    creates tfidf_dataset.csv from new_dataset.csv
    :return: -
    """
    # reads data/ArXiv_dataset.csv
    data_set = pd.read_csv("../data/new_dataset.csv",
                           sep=',',
                           header=0,
                           skiprows=0)

    # Find all categorical columns
    cols = data_set.columns
    num_cols = data_set._get_numeric_data().columns
    print(num_cols)

    categorical_col = list(set(cols) - set(num_cols))
    print(categorical_col)

    for x in categorical_col:
        #print(data_set[x].values)
        tfidf = TfidfVectorizer()
        result = tfidf.fit_transform(data_set[x].values.astype('U'))
        data_set[x] = list(result.toarray())

    print(data_set.head(5))

    # export csv
    data_set.to_csv('../data/tfidf_dataset.csv')



tf_idf()

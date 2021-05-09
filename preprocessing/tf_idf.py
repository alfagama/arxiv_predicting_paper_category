from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

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

    print(cols)
    # Remove all columns between column index 0 to 3
    data_set.drop(data_set.iloc[:, 0:3], inplace=True, axis=1)
    data_set.drop(['doi'], inplace=True, axis=1)
    cols = data_set.columns
    print(cols)

    categorical_col = list(set(cols) - set(num_cols))
    #Delete 'categories' column because we don't want to convert it to tf-idf
    categorical_col.remove('categories')
    print(categorical_col)

    #Perform tf-idf for all categorical columns
    for x in categorical_col:
        #print(data_set[x].values)
        tfidf = TfidfVectorizer()
        result = tfidf.fit_transform(data_set[x].values.astype('U'))
        # Save the result in the already existing column
        data_set[x] = list(result.toarray())

    print(data_set.head(1))

    # export csv
    data_set.to_csv('../data/tfidf_dataset.csv')

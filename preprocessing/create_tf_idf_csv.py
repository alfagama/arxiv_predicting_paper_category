from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from dataset.dataset_methods import get_preprocessed_dataset, split_df_train_test

# Options for pandas -----
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def tf_idf_vectorizer():
    """
    Firstly, it applies train test split and tf-idf. Secondly it creates csv files with train and test data
    :return: -
    """

    data = get_preprocessed_dataset()  # get preprcosessed dataset
    train_data, test_data = split_df_train_test(data)  # train test split

    text_col = 'pre_abstract' #tochange

    tfidf = TfidfVectorizer(max_features=3000)  # min_df=10, out of memory exception when max_features=4000
    train_data[text_col + 'tfidf'] = tfidf.fit_transform(train_data[text_col])
    test_data[text_col + 'tfidf'] = tfidf.transform(test_data[text_col])

    train_data.to_csv('../data/train_dataset_tfidf.csv')
    test_data.to_csv('../data/test_dataset_tfidf.csv')



tf_idf_vectorizer()



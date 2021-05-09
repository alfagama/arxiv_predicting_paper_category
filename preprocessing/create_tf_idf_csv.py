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

    data = get_preprocessed_dataset()  #.head(200)  # get preprcosessed dataset
    train_data, test_data = split_df_train_test(data)  # train test split

    text_col = 'concatenation'  # tochange
    print(train_data[text_col].values.astype('U'))
    tfidf = TfidfVectorizer(max_features=10000)  # min_df=10, out of memory exception when max_features=4000
    train_encodings = tfidf.fit_transform(train_data[text_col].values.astype('U'))
    train_data[text_col + '_tfidf'] = list(train_encodings.toarray())  # Save the result in the new column

    test_encodings = tfidf.transform(test_data[text_col].values.astype('U'))
    test_data[text_col + '_tfidf'] = list(test_encodings.toarray())  # Save the result in the new column

    train_data.to_csv('../data/train_dataset_tfidf.csv')
    test_data.to_csv('../data/test_dataset_tfidf.csv')


    # print tf idf scores (just for fun)
    df = pd.DataFrame(train_encodings[0].T.todense(), index=tfidf.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    print(df.head(25))

import pandas as pd
from collections import Counter
from imblearn.under_sampling import TomekLinks
from sklearn.feature_extraction.text import TfidfVectorizer
from dataset.dataset_methods import split_df_train_test
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, ADASYN, SMOTENC, RandomOverSampler
#from imblearn.ensemble import EasyEnsemble


def print_counter(y, message):
    counter = Counter(y)
    #print(y.size)
    print(message, counter)
    print(" ")


def read_and_sample_data():
    # reads data/category_columns_dataset.csv
    dataset = pd.read_csv("../data/category_columns_dataset.csv",
                          sep=',',
                          header=0,
                          skiprows=0)
    print('Dataset size is: ', dataset.size)
    # Sample my dataset
    my_dataset = dataset.sample(n=10000, replace=True, random_state=42)
    print('Sampled dataset size is: ', my_dataset.size)

    #return dataset;
    return my_dataset;


def tf_idf(dataset):

    train_data, test_data = split_df_train_test(dataset)  # train test split
    print(train_data.size)
    print(test_data.size)

    text_col = 'concatenation'  # tochange
    print(train_data[text_col].values.astype('U'))
    tfidf = TfidfVectorizer(max_features=1000)  # min_df=10, out of memory exception when max_features=4000
    train_encodings = tfidf.fit_transform(train_data[text_col].values.astype('U'))
    train_data[text_col + '_tfidf'] = list(train_encodings.toarray())  # Save the result in the new column
    print(train_data.size)
    test_encodings = tfidf.transform(test_data[text_col].values.astype('U'))
    test_data[text_col + '_tfidf'] = list(test_encodings.toarray())  # Save the result in the new column

    # train_data.to_csv('../data/train_dataset_tfidf.csv')
    # test_data.to_csv('../data/test_dataset_tfidf.csv')
    return train_data, test_data

def undersampling_methods(X,y):

    # Tomek Links
    # -------------------------------------------
    print_counter(y, 'Before Tomek Links undersampling')

    undersample = TomekLinks()
    X_TomekLinks, y_TomekLinks = undersample.fit_resample(X, y)
    print_counter(y_TomekLinks, 'After Tomek Links undersampling')
    # -------------------------------------------


    #ClusterCentroids
    # -------------------------------------------
    print_counter(y, 'Before Cluster Centroids undersampling')
    trans = ClusterCentroids(random_state=0)
    X_ClusterCentroids, y_ClusterCentroids = trans.fit_resample(X, y)

    # See the new class distribution
    print_counter(y_ClusterCentroids, 'After Cluster Centroids undersampling')
    # ta kanei undersampling OLA kai kataligoume na exoume apo kathe katigoria n_samples= number of minority class
    # -------------------------------------------

    """
    # EasyEnsemble
    # -------------------------------------------
    print_counter(y, 'Before EasyEnsemble undersampling')
    en = EasyEnsembleClassifier(random_state=42)
    X_ensemble, y_ensemble = en.fit_sample(X, y)
    print_counter(y_ensemble, 'After EasyEnsemble undersampling')
    # -------------------------------------------
    """

    return X_TomekLinks, y_TomekLinks, X_ClusterCentroids, y_ClusterCentroids; #, X_ensemble, y_ensemble;


def oversampling_methods(X,y):

    # SMOTE
    # -------------------------------------------
    print_counter(y, 'Before SMOTE undersampling')
    sm = SMOTE()
    X_smote, y_smote = sm.fit_resample(X, y)

    print_counter(y_smote, 'After SMOTE undersampling')
    # -------------------------------------------

    # Borderline SMOTE
    # -------------------------------------------
    print_counter(y, 'Before Borderline SMOTE undersampling')
    b_sm = BorderlineSMOTE()
    X_bsmote, y_bsmote = b_sm.fit_resample(X, y)

    print_counter(y_bsmote, 'After Borderline SMOTE undersampling')
    # -------------------------------------------

    return X_smote, y_smote, X_bsmote, y_bsmote


# Main
dataset = read_and_sample_data()
train_data, test_data = tf_idf(dataset)

y = train_data.categories
X = train_data.concatenation_tfidf.to_list()

X_TomekLinks, y_TomekLinks,X_ClusterCentroids, y_ClusterCentroids= undersampling_methods(X,y)
X_smote, y_smote, X_bsmote, y_bsmote = oversampling_methods(X,y)


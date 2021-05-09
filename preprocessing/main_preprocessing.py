from preprocessing.filter_dataset_based_on_category import *
from preprocessing.create_preprocessed_csv import *
from preprocessing.create_csv_with_categories_as_new_Columns import *
from preprocessing.create_tf_idf_csv import *

import os
import errno
import os.path


def run_dataset_creation_process():
    # check if directory /data exist. otherwise create it
    try:
        os.makedirs('../data')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # check for dataset from:
    # https://www.kaggle.com/Cornell-University/arxiv
    # if os.path.isfile('../data/ArXiv_dataset.csv'):
    #     # from preprocessing.filter_dataset_based_on_category
    #     create_csv()
    # if os.path.isfile('../data/dataset.csv'):
    #     # from preprocessing.create_preprocessed_csv
    #     preprocess()
    if os.path.isfile('../data/preprocessed_conc_dataset.csv'):
        # from preprocessing.create_csv_with_categories_as_new_Columns
        create_csv_with_categories_as_new_Columns()
    if os.path.isfile('../data/category_columns_dataset.csv'):
        print("...")
        # from preprocessing.create_tf_idf_csv
        # tf_idf_vectorizer()
        # from preprocessing.tf_idf
        # tf_idf()


run_dataset_creation_process()

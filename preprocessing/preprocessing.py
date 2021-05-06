import pandas as pd
import re
import nltk
import demoji
from datetime import datetime
from collections import Counter
from pymongo import MongoClient
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from preprocessing.dictionaries import *

nltk.download('wordnet')

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

    # df.to_csv('../data/preprocessed_dataset.csv')


preprocess()

import pandas as pd
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from preprocessing.dictionaries import *

# download wordnet from nltk library
nltk.download('wordnet')

# Options for pandas -----
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# stopwords in egnlish
eng_stopwords = stopwords.words('english')


def clear_column(row):
    """
    preprocess dataframe's rows
    :param row: run for each dataframe's row (dataframe row of X column)
    :return: preprocessed row (dataframe row of X column)
    """
    #   We modified the contractions & split them into tokens
    row = [get_contractions(word) for word in row.split()]

    #   We removed the hashtag symbol and its content (e.g., #COVID19), @users, and URLs from the messages because the
    #       hashtag symbols or the URLs did not contribute to the message analysis.
    row = [token for token in row if not token.startswith('http')
                        if not token.startswith('#') if not token.startswith('@') if
                        not token.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))]

    #   We removed all non-English characters (non-ASCII characters) because the study focused on the analysis of
    #       messages in English. (and numbers.)
    row = list(filter(lambda ele: re.search("[a-zA-Z\s]+", ele) is not None, row))

    #   We deTokenize here in order to use RE more efficinetly
    row = TreebankWordDetokenizer().detokenize(row)

    #   We use RE to remove any unwanted characters from the stings
    row = re.sub(r'[0-9]', ' ', row)
    row = re.sub(r'[!@#$%^&*(){}=_;:"“”‘’?.>,<`~.\\\[\]\-]', ' ', row)
    row = re.sub(r"[']", ' ', row)
    row = re.sub(r"[⁦⁩]", ' ', row)
    row = re.sub(r"[/]", ' ', row)
    row = re.sub(r"\t", " ", row)
    row = re.sub(r"'\s+\s+'", " ", row)
    row = re.sub(r" ️", "", row)
    row = re.sub(r'\b\w{1,1}\b', '', row)

    #   We tokenized again
    row = row.split()

    #   Remove unwanted tokens
    row = [x for x in row if x not in unwated_tokens]

    #   We removed stop_words
    final_stop_words = [x for x in eng_stopwords if x not in ok_stop_words]
    row = [w for w in row if w not in final_stop_words]

    #   We used WordNetLemmatizer from the nltk library as a final step
    lemmatizer = WordNetLemmatizer()
    # tokens_lemmatized = [lemmatizer.lemmatize(l) for l in tokens_no_stop_words]
    #   We also decided to use pos_tagging to enhance our lemmatization model
    tokens_lemmatized = []
    for word, tag in pos_tag(row):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        tokens_lemmatized.append(lemmatizer.lemmatize(word, pos))
    # return TreebankWordDetokenizer().detokenize(tokens_lemmatized)
    row = TreebankWordDetokenizer().detokenize(tokens_lemmatized)
    # print row to see results
    print(row)

    # return row
    return row


def preprocess():
    """
    initialize dataset from directory "../data/dataset.csv"
    and create preprocessed dataset "../data/preprocessed_dataset.csv"
    :return: -
    """
    # reads data/ArXiv_dataset.csv
    df = pd.read_csv("../data/dataset.csv",
                     sep=',',
                     header=0,
                     skiprows=0)
    # print(data_set.head(5))
    drop_columns = ['id']
    df = df.drop(drop_columns, axis=1)

    # # test run
    # df = df[:10]
    # print(df)

    #   We set all words to lower-case
    df.loc[:, "abstract"] = df.abstract.apply(lambda x: str.lower(x))
    df.loc[:, "categories"] = df.categories.apply(lambda x: str.lower(x))
    df.loc[:, "title"] = df.title.apply(lambda x: str.lower(x))
    # print(df.head(5))

    # call clear_column method for abstract
    df['pre_abstract'] = df.apply(lambda x: clear_column(x.abstract), axis=1)
    # call clear_column method for title
    df['pre_title'] = df.apply(lambda x: clear_column(x.title), axis=1)
    # print(df_preprocessed.head(5))

    # create preprocessed_dataset.csv
    df.to_csv('../data/preprocessed_dataset.csv')


# call method to initialize the pre-processing process
preprocess()

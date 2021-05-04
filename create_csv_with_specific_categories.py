import pandas as pd


def create_csv_with_specific_categories():
    """
    Creates a csv file containing only categories: cs, math, eess and stat
    :param -
    :return: -
    """

    data_set = pd.read_csv("data/ArXiv_dataset.csv",
                           sep=',',
                           header=0,
                           skiprows=0)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    drop_columns = ['authors_parsed', 'comments', 'id', 'journal-ref', 'license', 'report-no', 'submitter']
    data_set = data_set.drop(drop_columns, axis=1)
    print(data_set.head(5))

    patternToKeep = "(^|\W)cs\.+|math\.+|eess\.+|stat\.+"
    filter = data_set['categories'].str.contains(patternToKeep)
    data_set = data_set[filter]

    patternToDel = "(^|\W)physics+|cond-mat+|astro-ph+|hep-ph+|alg-geom+|hep-th+|quant-ph+|cmp-lg+|q-alg+|nlin+|gr-qc+|dg-ga+|econ+|q-fin+|q-bio+|hep+|econ+"
    filter = data_set['categories'].str.contains(patternToDel)
    data_set = data_set[~filter]


    print(data_set['categories'].value_counts(ascending=True))
    # data_set.categories.str.get_dummies(sep=' ').sum()

    data_set.to_csv('data/arxiv-cs-math-eess-stat.csv', index=False)
    print(len(data_set.index))



create_csv_with_specific_categories()



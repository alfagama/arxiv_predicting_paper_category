import pandas as pd

# Options for pandas -----
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def create_csv():
    """
    creates cs+math+eess+stat.csv from ArXiv_dataset.csv
    :return: -
    """
    # reads data/ArXiv_dataset.csv
    data_set = pd.read_csv("../data/ArXiv_dataset.csv",
                           sep=',',
                           header=0,
                           skiprows=0)

    # Drop columns
    drop_columns = ['authors_parsed', 'comments', 'id', 'journal-ref', 'license', 'report-no', 'submitter']
    data_set = data_set.drop(drop_columns, axis=1)
    print(data_set.head(5))

    # define pattern for subjects to keep
    pattern_to_keep = "(^|\W)cs\.+|math\.+|eess\.+|stat\.+"
    # create filter
    df_filter = data_set['categories'].str.contains(pattern_to_keep)
    # apply filter
    data_set = data_set[df_filter]

    # define pattern for subjects to delete
    pattern_to_del = "(^|\W)physics+|cond-mat+|astro-ph+|hep-ph+|alg-geom+|hep-th+|quant-ph+|cmp-lg+|q-alg+|nlin+|gr" \
                     "-qc+|dg-ga+|econ+|q-fin+|q-bio+|hep+|econ+ "
    # create filter
    df_del_filter = data_set['categories'].str.contains(pattern_to_del)
    # apply filter
    data_set = data_set[~df_del_filter]

    # keep only multi-label rows (>=2 categories)
    keep_multilabel = data_set['categories'].str.contains(' ')
    data_set = data_set[keep_multilabel]

    # -------------------------------------
    # print final data_set rows by category
    # -----
    # sort categories
    data_set['categories'] = data_set['categories'].apply(lambda x: ' '.join(sorted(x.split())))

    # group categories
    data_set = data_set.groupby('categories').filter(lambda x: len(x) > 1000)

    # print .value_counts of final dataset
    print(data_set['categories'].value_counts(ascending=True))
    # -------------------------------------

    # keep only if category starts wiht cs.
    # data_set = data_set['categories'].map(lambda x: x.startswith('cs.'))
    data_set = data_set[data_set['categories'].str.match('cs.')]
    # dataFrame[dataFrame['column name'].str.match('string')]
    print(data_set['categories'].value_counts(ascending=True))

    # export csv
    data_set.to_csv('../data/dataset.csv')

    print(data_set.head(5))

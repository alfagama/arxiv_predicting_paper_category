#pip install imbalanced-learn

#import imblearn
from collections import Counter
import pandas as pd

# Options for pandas -----
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def create_csv_with_categories_as_new_Columns():
    """
    creates cs+math+eess+stat.csv from ArXiv_dataset.csv
    :return: -
    """
    # reads data/ArXiv_dataset.csv
    data_set = pd.read_csv("../data/preprocessed_dataset.csv",
                           sep=',',
                           header=0,
                           skiprows=0)

    # Drop columns
    drop_columns = ['authors_parsed', 'comments', 'journal-ref', 'license', 'report-no', 'submitter']
    data_set = data_set.drop(drop_columns, axis=1)
    print(data_set.head(5))

    # define pattern for subjects to keep
    pattern_to_keep = "(^|\W)cs\.+|math\.+|eess\.+|stat\.+"
    # create filter
    df_filter = data_set['categories'].str.contains(pattern_to_keep)
    # apply filter
    data_set = data_set[df_filter]
    print(data_set.head(5))

    # define pattern for subjects to delete
    pattern_to_del = "(^|\W)physics+|cond-mat+|astro-ph+|hep-ph+|alg-geom+|hep-th+|quant-ph+|cmp-lg+|q-alg+|nlin+|gr" \
                     "-qc+|dg-ga+|econ+|q-fin+|q-bio+|hep+|econ+ "
    # create filter
    df_del_filter = data_set['categories'].str.contains(pattern_to_del)
    # apply filter
    data_set = data_set[~df_del_filter]
    print(data_set.head(5))

    # keep only multi-label rows (>=2 categories)
    keep_multilabel = data_set['categories'].str.contains(' ')
    data_set = data_set[keep_multilabel]

    col_one_list = data_set['categories'].tolist()

    unique_list = []
    category_Names = []

    for x in col_one_list:
        if x not in unique_list:
            unique_list.append(x)

    for categories in unique_list:
        split_categories = categories.split(" ")
        for category in split_categories:
            if category not in category_Names:
                category_Names.append(category)

    print('Category names are:')
    print(category_Names)

    for x in category_Names:
        contains = data_set['categories'].str.contains(x)
        print(contains)
        data_set[x] = contains.astype(int)
    print(data_set.head(5))

    # export csv
    data_set.to_csv('../data/new_dataset.csv')

    print(data_set.head(5))

    """
    # Create a counter to check the number of examples per category
    for x in category_Names:
        print(Counter(data_set[x]))
    """


create_csv_with_categories_as_new_Columns()
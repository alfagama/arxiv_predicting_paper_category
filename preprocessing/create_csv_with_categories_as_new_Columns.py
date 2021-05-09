import pandas as pd

# Options for pandas -----
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def create_csv_with_categories_as_new_Columns():
    """
    creates category_columns_dataset.csv from preprocessed_dataset.csv
    :return: -
    """
    # reads data/ArXiv_dataset.csv
    data_set = pd.read_csv("../data/preprocessed_dataset.csv",
                           sep=',',
                           header=0,
                           skiprows=0)

    print(data_set.head(5))

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
        data_set[x] = contains.astype(int)

    # export csv
    data_set.to_csv('../data/category_columns_dataset.csv')
    print(data_set.head(5))

create_csv_with_categories_as_new_Columns()
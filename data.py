import pandas as pd

def get_data():
    data_set = pd.read_csv("data/ArXiv_dataset.csv",
                           sep=',',
                           header=0,
                           skiprows=0)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    drop_columns = ['authors_parsed', 'comments', 'id', 'journal-ref', 'license', 'report-no', 'submitter']
    data_set = data_set.drop(drop_columns, axis=1)
    print(data_set.head(5))

    print(data_set['categories'].value_counts(ascending=True))
    # data_set.categories.str.get_dummies(sep=' ').sum()

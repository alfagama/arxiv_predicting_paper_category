import pandas as pd

# Options for pandas -----
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def create_multi_label_csv():
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
    drop_columns = ['authors_parsed', 'comments', 'id', 'journal-ref', 'license', 'report-no', 'submitter', 'authors', 'doi', 'versions', 'update_date', '_id']
    data_set = data_set.drop(drop_columns, axis=1)
    print(data_set.head(5))

    # delete rows that have more than one category
    df = data_set[data_set['categories'].apply(lambda x: True if len(x.split()) == 1 else False)]

    #print(df.head(30))
    #print(df.shape[0])

    # define pattern for subjects to keep
    pattern_to_keep = "(^|\W)cs\.+"  # "(^|\W)cs\.+|math\.+|eess\.+|stat\.+"
    # create filter
    df_filter = df['categories'].str.contains(pattern_to_keep)
    # apply filter
    df = df[df_filter]

    print(df['categories'].head(100))
    print(df.shape[0])

    # export csv
    df.head(30000).to_csv('../data/multi_class/multi_class_dataset_small.csv', index=False)



if __name__ == '__main__':
    create_multi_label_csv()
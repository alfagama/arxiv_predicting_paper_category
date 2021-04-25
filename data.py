import pandas as pd

def get_data():
    data_set = pd.read_csv("data/ArXiv_dataset.csv",
                           sep=',',
                           header=0,
                           skiprows=0)
    pd.set_option('display.max_columns', None)
    print(data_set.head(5))

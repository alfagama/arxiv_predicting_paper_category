"""
DISCLAIMER:
To whomever that may stumble upon this,
..I am terribly sorry!!
It was written at 01:00 am with a strict deadline!.
The important thing is that it works!?
So we get the results, in a nightmare-ish way..
:)
"""
from Imbalanced.imbalanced_methods import *

# Options for pandas -----
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def create_new_columns_from_df(df):
    """
    creates new columns from df
    :return: -
    """
    # print(df.head(5))
    col_one_list = df['categories'].tolist()
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

    # print('Category names are:')
    # print(category_Names)

    for x in category_Names:
        contains = df['categories'].str.contains(x)
        df[x] = contains.astype(int)

    return df


# ----------------------------------------------------------
if __name__ == '__main__':
    df = pd.read_csv("../data/preprocessed_conc_dataset.csv",
                     sep=',',
                     header=0,
                     skiprows=0)
    # df = df.sample(n=1000, replace=True, random_state=42)

    print("df length: ", len(df))

    # train test split
    train, test = train_test_split(df, random_state=42, test_size=0.30, shuffle=True)

    # ----------------------------------------------------------
    # Tf idf vectorizer
    # ----------------------------------------------------------
    train_text = train['concatenation'].values.astype('U')
    test_text = test['concatenation'].values.astype('U')

    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1, 3), norm='l2',
                                 max_features=10000)
    vectorizer.fit(train_text)
    x_train = vectorizer.transform(train_text)  # .toarray()
    x_test = vectorizer.transform(test_text)  # .toarray()
    # ----------------------------------------------------------

    y_train = train['categories']

    # ----------------------------------------------------------
    # UNDERSAMPLING
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # x_train
    # ----------------------------------------------------------
    X_TomekLinks, y_TomekLinks, \
    X_ClusterCentroids, y_ClusterCentroids, \
    X_RUS, y_RUS \
        = undersampling_methods(x_train, y_train)
    # undersample = TomekLinks()
    # X_TomekLinks, y_TomekLinks = undersample.fit_resample(x_train, y_train)

    # ----------------------------------------------------------
    # y_train -- Tomek
    # ----------------------------------------------------------
    # Tomek
    y_train_Tomek_df = pd.DataFrame()
    y_train_Tomek_df['categories'] = y_TomekLinks
    y_train_Tomek = create_new_columns_from_df(y_train_Tomek_df)
    y_train_Tomek = y_train_Tomek.drop(labels=['categories'], axis=1)
    y_train_Tomek = y_train_Tomek.reindex(sorted(y_train_Tomek.columns), axis=1)
    print(y_train_Tomek.head(2))
    y_train_Tomek = y_train_Tomek.to_numpy()
    # ClusterCentroids
    y_train_CC_df = pd.DataFrame()
    y_train_CC_df['categories'] = y_ClusterCentroids
    y_train_CC = create_new_columns_from_df(y_train_CC_df)
    y_train_CC = y_train_CC.drop(labels=['categories'], axis=1)
    y_train_CC = y_train_CC.reindex(sorted(y_train_CC.columns), axis=1)
    y_train_CC = y_train_CC.to_numpy()
    # RUS
    y_train_RUS_df = pd.DataFrame()
    y_train_RUS_df['categories'] = y_RUS
    y_train_RUS = create_new_columns_from_df(y_train_RUS_df)
    y_train_RUS = y_train_RUS.drop(labels=['categories'], axis=1)
    y_train_RUS = y_train_RUS.reindex(sorted(y_train_RUS.columns), axis=1)
    y_train_RUS = y_train_RUS.to_numpy()

    # ----------------------------------------------------------
    # x_test
    # ----------------------------------------------------------
    # x_test = x_test

    # ----------------------------------------------------------
    # y_test
    # ----------------------------------------------------------
    y_test = create_new_columns_from_df(test)
    y_test = y_test.drop(labels=['concatenation',
                                 'categories',
                                 'abstract',
                                 'title',
                                 'Unnamed: 0'
                                 ],
                         axis=1)
    print(y_test.head(3))
    y_test = y_test.reindex(sorted(y_test.columns), axis=1)
    print(y_test.head(3))
    y_test = y_test.to_numpy()

    # Tomek MLC
    print("")
    print("TomekLinks")
    print("length of dataset: ", len(y_train_Tomek), " + ", len(y_test))
    LabelPowersetClassification(X_TomekLinks, x_test, y_train_Tomek, y_test)
    BinaryRelevanceClassification(X_TomekLinks, x_test, y_train_Tomek, y_test)
    ClassifierChainsClassification(X_TomekLinks, x_test, y_train_Tomek, y_test)
    MLkNNClassification(X_TomekLinks, x_test, y_train_Tomek, y_test)

    # ClusterCentroids MLC
    print("")
    print("ClusterCentroids")
    print("length of dataset: ", len(y_train_CC), " + ", len(y_test))
    LabelPowersetClassification(X_ClusterCentroids, x_test, y_train_CC, y_test)
    BinaryRelevanceClassification(X_ClusterCentroids, x_test, y_train_CC, y_test)
    ClassifierChainsClassification(X_ClusterCentroids, x_test, y_train_CC, y_test)
    MLkNNClassification(X_ClusterCentroids, x_test, y_train_CC, y_test)

    # RUS MLC
    print("")
    print("RUS")
    print("length of dataset: ", len(y_train_RUS), " + ", len(y_test))
    LabelPowersetClassification(X_RUS, x_test, y_train_RUS, y_test)
    BinaryRelevanceClassification(X_RUS, x_test, y_train_RUS, y_test)
    ClassifierChainsClassification(X_RUS, x_test, y_train_RUS, y_test)
    MLkNNClassification(X_RUS, x_test, y_train_RUS, y_test)

    # ----------------------------------------------------------
    # UNDERSAMPLING NEAR MISS
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # x_train
    # ----------------------------------------------------------
    X_nm, y_nm, \
    X_nm2, y_nm2, \
    X_nm3, y_nm3 \
        = undersampling_NearMiss_methods(x_train, y_train)

    # ----------------------------------------------------------
    # y_train -- NM1
    # ----------------------------------------------------------

    # NM1
    y_train_NM1_df = pd.DataFrame()
    y_train_NM1_df['categories'] = y_nm
    y_train_NM1 = create_new_columns_from_df(y_train_NM1_df)
    y_train_NM1 = y_train_NM1.drop(labels=['categories'], axis=1)
    y_train_NM1 = y_train_NM1.reindex(sorted(y_train_NM1.columns), axis=1)
    y_train_NM1 = y_train_NM1.to_numpy()

    # NM2
    y_train_NM2_df = pd.DataFrame()
    y_train_NM2_df['categories'] = y_nm2
    y_train_NM2 = create_new_columns_from_df(y_train_NM2_df)
    y_train_NM2 = y_train_NM2.drop(labels=['categories'], axis=1)
    y_train_NM2 = y_train_NM2.reindex(sorted(y_train_NM2.columns), axis=1)
    y_train_NM2 = y_train_NM2.to_numpy()

    # NM3
    y_train_NM3_df = pd.DataFrame()
    y_train_NM3_df['categories'] = y_nm3
    y_train_NM3 = create_new_columns_from_df(y_train_NM3_df)
    y_train_NM3 = y_train_NM3.drop(labels=['categories'], axis=1)
    y_train_NM3 = y_train_NM3.reindex(sorted(y_train_NM3.columns), axis=1)
    y_train_NM3 = y_train_NM3.to_numpy()

    # Tomek MLC
    print("")
    print("Near Miss 1")
    print("length of dataset: ", len(y_train_NM1), " + ", len(y_test))
    LabelPowersetClassification(X_nm, x_test, y_train_NM1, y_test)
    BinaryRelevanceClassification(X_nm, x_test, y_train_NM1, y_test)
    ClassifierChainsClassification(X_nm, x_test, y_train_NM1, y_test)
    MLkNNClassification(X_nm, x_test, y_train_NM1, y_test)

    # ClusterCentroids MLC
    print("")
    print("Near Miss 2")
    print("length of dataset: ", len(y_train_NM2), " + ", len(y_test))
    LabelPowersetClassification(X_nm2, x_test, y_train_NM2, y_test)
    BinaryRelevanceClassification(X_nm2, x_test, y_train_NM2, y_test)
    ClassifierChainsClassification(X_nm2, x_test, y_train_NM2, y_test)
    MLkNNClassification(X_nm2, x_test, y_train_NM2, y_test)

    # RUS MLC
    print("")
    print("Near Miss 3")
    print("length of dataset: ", len(y_train_NM3), " + ", len(y_test))
    LabelPowersetClassification(X_nm3, x_test, y_train_NM3, y_test)
    BinaryRelevanceClassification(X_nm3, x_test, y_train_NM3, y_test)
    ClassifierChainsClassification(X_nm3, x_test, y_train_NM3, y_test)
    MLkNNClassification(X_nm3, x_test, y_train_NM3, y_test)

    # ----------------------------------------------------------
    # COMBINATIONS
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # x_train
    # ----------------------------------------------------------
    X_smoteTomek, y_smoteTomek = combination_methods(x_train, y_train)

    # ----------------------------------------------------------
    # y_train -- NM1
    # ----------------------------------------------------------

    # y_smoteTomek
    y_ST_df = pd.DataFrame()
    y_ST_df['categories'] = y_smoteTomek
    y_ST = create_new_columns_from_df(y_ST_df)
    y_ST = y_ST.drop(labels=['categories'], axis=1)
    y_ST = y_ST.reindex(sorted(y_ST.columns), axis=1)
    y_ST = y_ST.to_numpy()

    # y_smoteTomek
    print("")
    print("y_smoteTomek")
    print("length of dataset: ", len(y_ST), " + ", len(y_test))
    LabelPowersetClassification(X_smoteTomek, x_test, y_ST, y_test)
    BinaryRelevanceClassification(X_smoteTomek, x_test, y_ST, y_test)
    ClassifierChainsClassification(X_smoteTomek, x_test, y_ST, y_test)
    MLkNNClassification(X_smoteTomek, x_test, y_ST, y_test)

    # ----------------------------------------------------------
    # OVERSAMPLING
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # x_train
    # ----------------------------------------------------------
    X_smote, y_smote, \
    X_bsmote, y_bsmote, \
    X_ROS, y_ROS \
        = oversampling_methods(x_train, y_train)

    # ----------------------------------------------------------
    # y_train -- NM1
    # ----------------------------------------------------------
    # SMOTE
    y_train_SMOTE_df = pd.DataFrame()
    y_train_SMOTE_df['categories'] = y_smote
    y_train_SMOTE = create_new_columns_from_df(y_train_SMOTE_df)
    y_train_SMOTE = y_train_SMOTE.drop(labels=['categories'], axis=1)
    y_train_SMOTE = y_train_SMOTE.reindex(sorted(y_train_SMOTE.columns), axis=1)
    y_train_SMOTE = y_train_SMOTE.to_numpy()

    # BSMOTE
    y_train_BSMOTE_df = pd.DataFrame()
    y_train_BSMOTE_df['categories'] = y_bsmote
    y_train_BSMOTE = create_new_columns_from_df(y_train_BSMOTE_df)
    y_train_BSMOTE = y_train_BSMOTE.drop(labels=['categories'], axis=1)
    y_train_BSMOTE = y_train_BSMOTE.reindex(sorted(y_train_BSMOTE.columns), axis=1)
    y_train_BSMOTE = y_train_BSMOTE.to_numpy()

    # ROS
    y_train_ROS_df = pd.DataFrame()
    y_train_ROS_df['categories'] = y_ROS
    y_train_ROS = create_new_columns_from_df(y_train_ROS_df)
    y_train_ROS = y_train_ROS.drop(labels=['categories'], axis=1)
    y_train_ROS = y_train_ROS.reindex(sorted(y_train_ROS.columns), axis=1)
    y_train_ROS = y_train_ROS.to_numpy()

    # SMOTE
    print("")
    print("SMOTE")
    print("length of dataset: ", len(y_train_SMOTE), " + ", len(y_test))
    LabelPowersetClassification(X_smote, x_test, y_train_SMOTE, y_test)
    BinaryRelevanceClassification(X_smote, x_test, y_train_SMOTE, y_test)
    ClassifierChainsClassification(X_smote, x_test, y_train_SMOTE, y_test)
    MLkNNClassification(X_smote, x_test, y_train_SMOTE, y_test)

    # BSMOTE
    print("")
    print("BSMOTE")
    print("length of dataset: ", len(y_train_BSMOTE), " + ", len(y_test))
    LabelPowersetClassification(X_bsmote, x_test, y_train_BSMOTE, y_test)
    BinaryRelevanceClassification(X_bsmote, x_test, y_train_BSMOTE, y_test)
    ClassifierChainsClassification(X_bsmote, x_test, y_train_BSMOTE, y_test)
    MLkNNClassification(X_bsmote, x_test, y_train_BSMOTE, y_test)

    # ROS
    print("")
    print("ROS")
    print("length of dataset: ", len(y_train_ROS), " + ", len(y_test))
    LabelPowersetClassification(X_ROS, x_test, y_train_ROS, y_test)
    BinaryRelevanceClassification(X_ROS, x_test, y_train_ROS, y_test)
    ClassifierChainsClassification(X_ROS, x_test, y_train_ROS, y_test)
    MLkNNClassification(X_ROS, x_test, y_train_ROS, y_test)
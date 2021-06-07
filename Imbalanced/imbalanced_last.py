import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from imblearn.under_sampling import ClusterCentroids, NearMiss, RandomUnderSampler, TomekLinks
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, ADASYN, SMOTENC, RandomOverSampler
from imblearn.combine import SMOTETomek
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# Options for pandas -----
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if __name__ == '__main__':

    models = [
        RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0),
        DecisionTreeClassifier(random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
        GaussianNB(),
        KNeighborsClassifier(),
    ]

    dataframe = pd.read_csv("../data/dataset_multiclass_preprocessed.csv",
                            sep=',',
                            header=0,
                            skiprows=0)
    dataframe = dataframe.sample(2000, random_state=0)

    # print(dataframe.head(2))
    print("df length: ", len(dataframe))
    print(dataframe.categories.unique())
    dataframe = dataframe[dataframe.categories != 'cs.hc']
    dataframe = dataframe[dataframe.categories != 'cs.si']
    print("df length: ", len(dataframe))
    print(dataframe.categories.unique())

    print(dataframe.groupby('categories').count())

    # X = df['concatenation']
    # y = df['categories']
    #
    # print(len(X))
    # print(len(y))
    #
    # print(X)
    # print(y)
    #
    # x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    #
    # # ----------------------------------------------------------
    # # Tf idf vectorizer
    # # ----------------------------------------------------------
    # train_text = x_train.values.astype('U')
    # test_text = x_test.values.astype('U')
    #
    # vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1, 3), norm='l2',
    #                              max_features=10000)
    # vectorizer.fit(train_text)
    # x_train = vectorizer.transform(train_text)  # .toarray()
    # x_test = vectorizer.transform(test_text)  # .toarray()
    #
    # print(x_train)

    # dataframe with factorize
    df = dataframe[['concatenation', 'categories']]
    # print(df.head(2))
    df['category_id'] = df['categories'].factorize()[0]
    print(df.head(2))

    # create a couple of dictionaries for future use
    category_id_df = df[['concatenation', 'category_id']].sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'concatenation']].values)

    # sho dataset plot
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle('ArXiv Papers - Dataset')
    df.groupby('categories').concatenation.count().plot.bar(ylim=0)
    plt.savefig('../data_pics/imbalanced_dataset.png')
    # plt.show()
    plt.close()

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',
                            ngram_range=(1, 3), stop_words='english', max_features=10000,
                            strip_accents='unicode', analyzer='word')
    features = tfidf.fit_transform(df.concatenation).toarray()
    labels = df.category_id
    print(features.shape)
    print(df.head(5))

    # # We can use sklearn.feature_selection.chi2 to find the terms
    # #     that are the most correlated with each of the products
    # N = 2
    # for concatenation, category_id in sorted(category_to_id.items()):
    #     features_chi2 = chi2(features, labels == category_id)
    #     indices = np.argsort(features_chi2[0])
    #     feature_names = np.array(tfidf.get_feature_names())[indices]
    #     unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    #     bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    #     print("# '{}':".format(concatenation))
    #     print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
    #     print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

    X_train, X_test, y_train, y_test = train_test_split(
        df['concatenation'], df['categories'], random_state=0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # cross-validation -> see best model -> run CV times
    # CV = 10
    # cv_df = pd.DataFrame(index=range(CV * len(models)))
    # entries = []
    # for model in models:
    #   model_name = model.__class__.__name__
    #   accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    #   for fold_idx, accuracy in enumerate(accuracies):
    #     entries.append((model_name, fold_idx, accuracy))
    # cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    #
    # sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    # sns.stripplot(x='model_name', y='accuracy', data=cv_df,
    #               size=8, jitter=True, edgecolor="gray", linewidth=2)
    # plt.xticks(rotation=90)
    # plt.gcf().subplots_adjust(bottom=0.25)
    # plt.savefig('../data_pics/imbalanced_cv.png')
    # plt.show()
    # plt.close()
    #
    # print(cv_df.groupby('model_name').accuracy.mean())

    # best model
    model = LinearSVC()

    ##############################################################
    # LOOP edw gia oles tis methodous imbalance
    # mia for kai olos o kwdikas pros ta mesa
    # epishs mporoume kai to panw plot mesa sth loop na bgazei kathe fora neo gia olous tous classifiers
    ##############################################################

    # tomekLinks = TomekLinks()
    # X_TomekLinks, y_TomekLinks = tomekLinks.fit_resample(features,labels)

    rus = RandomUnderSampler(random_state=777)
    X_RUS, y_RUS = rus.fit_resample(features, labels)

    print('lengths')
    print(len(labels))
    print(len(y_RUS))
    # X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
    X_train, X_test, y_train, y_test = train_test_split(X_RUS, y_RUS, test_size=0.33, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=df['categories'].unique()))

    # Define a result table as a DataFrame
    result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

    # Train the models and record the results
    for cls in models:
        print("models X")
        print(cls)
        print(X_train)
        model = cls.fit(X_train, y_train)
        if str(cls) != 'LinearSVC()':
            yproba = model.predict_proba(X_test)[::, 1]
        else:
            yproba = model.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, yproba)
        auc = roc_auc_score(y_test, yproba)

        result_table = result_table.append({'classifiers': cls.__class__.__name__,
                                            'fpr': fpr,
                                            'tpr': tpr,
                                            'auc': auc}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    fig = plt.figure(figsize=(8, 6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')
    plt.savefig('../data_pics/imbalanced_roc.png')
    plt.show()

    #####################################
    ## PRE REC CURVE plot for multiple classifiers
    #########################################

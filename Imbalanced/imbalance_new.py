import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import ClusterCentroids, NearMiss, RandomUnderSampler, TomekLinks
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_curve, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import time

def plot_curves(imb,result_table, prec_rec_table):
    # ROC AUC CURVE
    result_table.set_index('classifiers', inplace=True)
    print(result_table.head(2))

    fig = plt.figure(figsize=(8, 6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')
    #plt.savefig('../data_pics/'+imb+'_roc.png')
    plt.show()

    #PRECISION RECALL CURVE

    # Set name of the classifiers as index labels
    prec_rec_table.set_index('classifiers', inplace=True)

    fig = plt.figure(figsize=(8, 6))

    for i in prec_rec_table.index:
        plt.plot(prec_rec_table.loc[i]['rec'],
                 prec_rec_table.loc[i]['pr'],
                 label="{}, AP={:.3f}".format(i, prec_rec_table.loc[i]['ap']))

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Precision", fontsize=15)

    plt.title('Precision vs Recall curve', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')
    #plt.savefig('../data_pics/'+imb+'_prec_recall.png')
    plt.show()

def run_pipeline(imb,result_table,prec_rec_table):
    count_vect = CountVectorizer(max_features=25000)

    ##Create the models
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0),
        DecisionTreeClassifier(random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0, max_iter=500),
        KNeighborsClassifier(),
    ]

    for model in models:
        pipe = make_pipeline(count_vect,imb,model)
        start = time.time()

        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        if str(model) != 'LinearSVC()':
            y_pred_prob = pipe.predict_proba(x_test)[:, 1]
        else:
            y_pred_prob = pipe.predict(x_test)
        end = time.time()

        print('Results for ' + str(model))
        print('f1 score: ', f1_score(y_test, y_pred, average='macro'))
        print('balanced accuracy score: ', balanced_accuracy_score(y_test, y_pred))
        print("Time elapsed: ", end - start)  # CPU seconds elapsed (floating point)

        # print('ROC AUC score: ', roc_auc_score(y_test, y_pred_prob, multi_class='ovr'))
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)
        pr, rc, _ = precision_recall_curve(y_test, y_pred_prob)
        average_precision = average_precision_score(y_test, y_pred_prob)
        print('Roc auc score: ', auc)
        print('Precision recall score: ', average_precision)
        print()

        prec_rec_table = prec_rec_table.append({'classifiers': model.__class__.__name__,
                                            'pr': pr,
                                            'rec': rc,
                                            'ap': average_precision}, ignore_index=True)

        result_table = result_table.append({'classifiers': model.__class__.__name__,
                                            'fpr': fpr,
                                            'tpr': tpr,
                                            'auc': auc}, ignore_index=True)

    return result_table,prec_rec_table



if __name__ == '__main__':

    # Define the roc and prec-recall tables as a DataFrames
    roc_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
    prec_rec_table = pd.DataFrame(columns=['classifiers', 'pr', 'rec', 'ap'])

    df = pd.read_csv("../data/dataset_multiclass_preprocessed.csv",
                     sep=',',
                     header=0,
                     skiprows=0)
    #df = df.sample(1000, random_state=0)

    # print(dataframe.head(2))
    print("df length: ", len(df))
    print(df.categories.unique())
    dataframe = df[df.categories != 'cs.hc']
    dataframe = dataframe[dataframe.categories != 'cs.si']
    print("df length: ", len(dataframe))
    print(dataframe.categories.unique())

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
    #plt.savefig('../data_pics/imbalanced_dataset.png')
    # plt.show()
    plt.close()


    X = df['concatenation']
    y = df['category_id']


    n_classes = len(set(df.categories))
    print('Number of classes is: ', n_classes)

    df_classes = df['categories'].value_counts().rename_axis('Categories').reset_index(name='Row count')
    print(df_classes)
    ax = df_classes.plot.bar(x='Categories', y='Row count', rot=0, title='Categories distribution')
    ax.figure.savefig('imbalanced_distr.png')


    heights = df_classes['Row count']
    bars = df_classes['Categories']
    y_pos = range(len(bars))
    plt.bar(y_pos, heights)
    plt.xticks(y_pos, bars, rotation=90)
    plt.ylabel('Row count')
    plt.legend().remove()
    plt.grid(False)
    plt.gcf().subplots_adjust(bottom=0.25)
    #plt.savefig('synonym_distr.png')
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    imb_methods = [
        RandomUnderSampler(random_state=777),
        NearMiss(sampling_strategy='not minority', version=1, n_neighbors=1),
        NearMiss(sampling_strategy='not minority', version=2, n_neighbors=1),
        NearMiss(sampling_strategy='not minority', version=3, n_neighbors_ver3=4),
        SMOTE(),
        BorderlineSMOTE(),
        RandomOverSampler(random_state=777),
        SMOTETomek(tomek=TomekLinks(sampling_strategy='majority')),
        TomekLinks(),
        ClusterCentroids(random_state=0)
    ]

    for imb in imb_methods:
        print('--------------'+ str(imb)+'--------------')
        roc_table,prec_rec_table = run_pipeline(imb,roc_table,prec_rec_table)
        print('----------------------------')
        print(roc_table)
        plot_curves(str(imb),roc_table, prec_rec_table)
        roc_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
        prec_rec_table = pd.DataFrame(columns=['classifiers', 'pr', 'rec', 'ap'])











import pandas as pd
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
#from Imbalanced.imbalanced_methods import *
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import ClusterCentroids, NearMiss, RandomUnderSampler, TomekLinks
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, ADASYN, SMOTENC, RandomOverSampler
from imblearn.combine import SMOTETomek
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from yellowbrick.classifier import PrecisionRecallCurve
import numpy as np
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

def imbalanced_methods(model, modelName):
    tf_idf = TfidfVectorizer(max_features=10000)
    """
    ##########################################################
    #Undersampling

    # Tomek Links
    # -------------------------------------------
    tomekLinks = TomekLinks()
    print('Results for Tomek Links undersampling')
    run_pipeline(tf_idf, tomekLinks, model,'TomekLinks',modelName,result_table)
    # -------------------------------------------

    # ClusterCentroids
    # -------------------------------------------
    clusterCent = ClusterCentroids(random_state=0)
    print('Results for Cluster Centroids undersampling')
    run_pipeline(tf_idf, clusterCent, model,'ClusterCentroids',modelName,result_table)
    # ta kanei undersampling OLA kai kataligoume na exoume apo kathe katigoria n_samples= number of minority class
    # -------------------------------------------
    
    """

    # Random Under Sampler
    # -------------------------------------------
    rus = RandomUnderSampler(random_state=777)
    print('Results for Random Under Sampler undersampling')
    run_pipeline(tf_idf, rus, model,'RandomUnderSampler',modelName)
    # -------------------------------------------

    # undersampling_NearMiss_methods
    nm1 = NearMiss(sampling_strategy='not minority', version=1, n_neighbors=1)  # random_state=777,
    print('Results for NearMiss-1 undersampling')
    pipe = run_pipeline(tf_idf, nm1, model,'NearMiss1',modelName)

    nm2 = NearMiss(sampling_strategy='not minority', version=2, n_neighbors=1)
    print('Results for NearMiss-2 undersampling')
    run_pipeline(tf_idf, nm2, model,'NearMiss2',modelName)

    nm3 = NearMiss(sampling_strategy='not minority', version=3, n_neighbors_ver3=4)
    print('Results for NearMiss-3 undersampling')
    run_pipeline(tf_idf, nm3, model,'NearMiss3',modelName)

    ##########################################################
    # Oversampling

    # SMOTE
    # -------------------------------------------
    smote = SMOTE()
    print('Results for SMOTE oversampling')
    run_pipeline(tf_idf, smote, model,'SMOTE',modelName)
    # -------------------------------------------

    # Borderline SMOTE
    # -------------------------------------------
    b_sm = BorderlineSMOTE()
    print('Results for Borderline SMOTE oversampling')
    run_pipeline(tf_idf, b_sm, model,'BorderlineSMOTE',modelName)
    # -------------------------------------------

    # Random Over Sampler
    # -------------------------------------------
    ros = RandomOverSampler(random_state=777)
    print('Results for Random Over Sampler oversampling')
    run_pipeline(tf_idf, ros, model,'RandomOverSampler',modelName)
    # -------------------------------------------

    ##########################################################
    # Combination

    # SMOTE + Tomek Links
    # -------------------------------------------
    smTomek = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
    print('Results for SMOTE + Tomek Links')
    run_pipeline(tf_idf, smTomek, model,'SMOTETomek',modelName)

    return result_table

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
    plt.savefig('../data_pics/'+imb+'_roc.png')
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
    plt.savefig('../data_pics/'+imb+'_prec_recall.png')
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
        pipe = make_pipeline(count_vect, imb, model)

        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        if str(model) != 'LinearSVC()':
            y_pred_prob = pipe.predict_proba(x_test)[:, 1]
        else:
            y_pred_prob = pipe.predict(x_test)
        # y_pred_prob = pipe.predict_proba(x_test)

        print('Results for ' + str(model))
        print('f1 score: ', f1_score(y_test, y_pred, average='macro'))
        print('balanced accuracy score: ', balanced_accuracy_score(y_test, y_pred))
        # print('ROC AUC score: ', roc_auc_score(y_test, y_pred_prob, multi_class='ovr'))
        print()
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)
        pr, rc, _ = precision_recall_curve(y_test, y_pred_prob)
        average_precision = average_precision_score(y_test, y_pred_prob)

        prec_rec_table = prec_rec_table.append({'classifiers': model.__class__.__name__,
                                            'pr': pr,
                                            'rec': rc,
                                            'ap': average_precision}, ignore_index=True)

        result_table = result_table.append({'classifiers': model.__class__.__name__,
                                            'fpr': fpr,
                                            'tpr': tpr,
                                            'auc': auc}, ignore_index=True)

    """
    pipe = Pipeline([
        ('tf_idf', tf_idf),
        #('count_vec', count_vect),
        # ('svd', svd),
        ('imbalanced_method', imb),
        ('algorithm', model)
    ])
    """
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
    plt.savefig('../data_pics/imbalanced_dataset.png')
    # plt.show()
    plt.close()


    X = df['concatenation']
    y = df['category_id']


    n_classes = len(set(df.categories))
    print('Number of classes is: ', n_classes)

    df_classes = df['categories'].value_counts().rename_axis('Categories').reset_index(name='Row count')
    print(df_classes)
    ax = df_classes.plot.bar(x='Categories', y='Row count', rot=0, title='Categories distribution')
    ax.figure.savefig('categories_distribution.png')

    #y = label_binarize(y, classes=[*range(n_classes)])
    #print(y)

    #lb = preprocessing.LabelBinarizer()
    #y = lb.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    """
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
    """
    imb_methods = [
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

    """
    lr = LogisticRegression(random_state=0, max_iter=500)
    print('--------------Results for Logistic Regression--------------')
    result_table = imbalanced_methods(lr,'LogisticRegression',result_table)
    print('----------------------------')

    naive_bayes = MultinomialNB()
    print('--------------Results for Multinomial Naive Bayes--------------')
    result_table = imbalanced_methods(naive_bayes, 'MultinomialNB',result_table)
    print('----------------------------')

    #clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=50,
    #                                                 max_depth=3,
    #                                                 random_state=0))
    random_forest = RandomForestClassifier(n_estimators=1000, random_state=42)
    print('--------------Results for Random Forest Classifier-------------')
    result_table = imbalanced_methods(random_forest, 'RandomForestClassifier',result_table)
    print('----------------------------')
    """



#svd = TruncatedSVD(n_components=5, n_iter=100, random_state=42)













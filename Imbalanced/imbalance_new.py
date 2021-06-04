from Imbalanced.imbalanced_methods import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import ClusterCentroids, NearMiss, RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, ADASYN, SMOTENC, RandomOverSampler
from imblearn.combine import SMOTETomek
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, balanced_accuracy_score, roc_auc_score
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def imbalanced_methods(model):
    tf_idf = TfidfVectorizer(max_features=10000)

    ##########################################################
    #Undersampling

    # Tomek Links
    # -------------------------------------------
    tomekLinks = TomekLinks()
    print('Results for Tomek Links undersampling')
    run_pipeline(tf_idf, tomekLinks, model)
    # -------------------------------------------

    # ClusterCentroids
    # -------------------------------------------
    clusterCent = ClusterCentroids(random_state=0)
    print('Results for Cluster Centroids undersampling')
    run_pipeline(tf_idf, clusterCent, model)
    # ta kanei undersampling OLA kai kataligoume na exoume apo kathe katigoria n_samples= number of minority class
    # -------------------------------------------

    # Random Under Sampler
    # -------------------------------------------
    rus = RandomUnderSampler(random_state=777)
    print('Results for Random Under Sampler undersampling')
    run_pipeline(tf_idf, rus, model)
    # -------------------------------------------

    # undersampling_NearMiss_methods
    nm1 = NearMiss(sampling_strategy='not minority', version=1, n_neighbors=1)  # random_state=777,
    print('Results for NearMiss-1 undersampling')
    pipe = run_pipeline(tf_idf, nm1, model)

    nm2 = NearMiss(sampling_strategy='not minority', version=2, n_neighbors=1)
    print('Results for NearMiss-2 undersampling')
    run_pipeline(tf_idf, nm2, model)

    nm3 = NearMiss(sampling_strategy='not minority', version=3, n_neighbors_ver3=4)
    print('Results for NearMiss-3 undersampling')
    run_pipeline(tf_idf, nm3, model)

    ##########################################################
    # Oversampling

    # SMOTE
    # -------------------------------------------
    smote = SMOTE()
    print('Results for SMOTE oversampling')
    run_pipeline(tf_idf, smote, model)
    # -------------------------------------------

    # Borderline SMOTE
    # -------------------------------------------
    b_sm = BorderlineSMOTE()
    print('Results for Borderline SMOTE oversampling')
    run_pipeline(tf_idf, b_sm, model)
    # -------------------------------------------

    # Random Over Sampler
    # -------------------------------------------
    ros = RandomOverSampler(random_state=777)
    print('Results for Random Over Sampler oversampling')
    run_pipeline(tf_idf, ros, model)
    # -------------------------------------------


def run_pipeline(tf_idf,imb,model):

    pipe = Pipeline([
        ('tf_idf', tf_idf),
        # ('svd', svd),
        ('imbalanced_method', imb),
        ('algorithm', model)
    ])
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    y_pred_prob = pipe.predict_proba(x_test)

    print('f1 score: ', f1_score(y_test, y_pred, average='macro'))
    print('balanced accuracy score: ', balanced_accuracy_score(y_test, y_pred))
    print('ROC AUC score: ', roc_auc_score(y_test, y_pred_prob, multi_class='ovr'))
    print()
    #metrics.plot_roc_curve(pipe, x_test, y_test)
    #plt.show()



if __name__ == '__main__':
    df = pd.read_csv("../data/preprocessed_conc_dataset.csv",
                     sep=',',
                     header=0,
                     skiprows=0)
    df = df.sample(2000, random_state=0)

    print("df length: ", len(df))

    X = df['concatenation']
    y = df['categories']

    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)

    ##Create the models
    lr = LogisticRegression(random_state=0, max_iter=500)
    imbalanced_methods(lr)

    naive_bayes = MultinomialNB()
    imbalanced_methods(naive_bayes)

    random_forest = RandomForestClassifier(n_estimators=1000, random_state=42)
    imbalanced_methods(random_forest)



#svd = TruncatedSVD(n_components=5, n_iter=100, random_state=42)













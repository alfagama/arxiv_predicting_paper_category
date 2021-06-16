from sklearn.model_selection import train_test_split
import shap
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


if __name__ == '__main__':
    data = pd.read_csv("../data/multi_class/dataset_multiclass_preprocessed_big.csv")#.head(10000)

    data = data.loc[(data['categories'] == 'cs.gl') | (data['categories'] == 'cs.os') |
                (data['categories'] == 'cs.ms') | (data['categories'] == 'cs.pf') |
                (data['categories'] == 'cs.ma') | (data['categories'] == 'cs.ar') |
                (data['categories'] == 'cs.si') | (data['categories'] == 'cs.ir') |
                (data['categories'] == 'cs.ce') | (data['categories'] == 'cs.sd') |
                (data['categories'] == 'cs.pl')]
    #print(data)

    X = data[['concatenation']]
    y = data[['categories']]

    # Τrain-test Split
    # -----------------------
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(x_train.shape[0])
    print(y_train.shape[0])
    # -----------------------

    # Count Vectorizer
    # -----------------------
    vectorizer = CountVectorizer(analyzer='word')
    x_train = vectorizer.fit_transform(x_train['concatenation'])
    x_test = vectorizer.transform(x_test['concatenation'])
    # -----------------------

    # Linear SVC Classification
    # -----------------------
    model = LinearSVC()
    model.fit(x_train, y_train)
    # -----------------------

    # Predict on test data
    # -----------------------
    pred = model.predict(x_test)
    # -----------------------

    # Evaluation Metrics
    # -----------------------
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='weighted')
    recall = recall_score(y_test, pred, average='weighted')
    f1 = f1_score(y_test, pred, average='weighted')
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f\n" % (accuracy, precision, recall, f1))
    # -----------------------

    # SHAP Explainer
    # -----------------------
    explainer = shap.LinearExplainer(model, x_train, feature_dependence="independent")
    shap_values = explainer.shap_values(x_test)
    X_test_array = x_test.toarray() # we need to pass a dense version for the plotting functions

    fig = shap.summary_plot(shap_values, X_test_array, feature_names=vectorizer.get_feature_names(), class_names=model.classes_, show=False)
    plt.tight_layout()
    plt.savefig('shap_LinearSVC.png')
    # -----------------------

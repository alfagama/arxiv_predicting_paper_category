from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn import metrics
import eli5
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss


def print_evaluation_report(pipeline):
    y_actuals = y_test['categories']
    y_preds = pipeline.predict(x_test['concatenation'])
    report = metrics.classification_report(y_actuals, y_preds)

    print(report)
    print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_actuals, y_preds)))
    print("F1 score = ", f1_score(y_test, y_preds, average="micro"))
    print("Hamming loss = ", hamming_loss(y_test, y_preds))


def print_coefficients():

    for i, tag in enumerate(clf.classes_):
        coefficients = clf.coef_[i]
        weights = list(zip(vec.get_feature_names(), coefficients))
        print('Tag:', tag)
        print('Most Positive Coefficients:')
        print(sorted(weights, key=lambda x: -x[1])[:10])
        print('Most Negative Coefficients:')
        print(sorted(weights, key=lambda x: x[1])[:10])
        print("--------------------------------------")


def create_html_visualization():
    html_obj = eli5.show_weights(clf, vec=vec, top=20)

    # Write html object to a file (adjust file path; Windows path is used here)
    with open('weights_CountVectorizer_5000_LogisticRegression2.html', 'wb') as f:
        f.write(html_obj.data.encode("UTF-8"))


def misclassified_examples(pipeline):
    y_preds = pipeline.predict(data['concatenation'])
    data['predicted_label'] = y_preds
    misclassified_examples = data[(data['categories'] != data['predicted_label']) & (data['categories'] == 'cs.ai')]  # & (
    # data['predicted_label'] == 'cs.ar')]

    print("Actual Target Value : ", misclassified_examples['predicted_label'].values[1])
    print("Model Prediction : ", misclassified_examples['categories'].values[1])
    html_obj = eli5.show_prediction(clf, misclassified_examples['concatenation'].values[1], vec=vec)

    # Write html object to a file (adjust file path; Windows path is used here)
    with open('missclassification_CountVectorizer_5000_LogisticRegression2.html', 'wb') as f:
        f.write(html_obj.data.encode("UTF-8"))


if __name__ == '__main__':
    data = pd.read_csv("../data/multi_class/dataset_multiclass_preprocessed_big.csv")
    print(data)

    # Creating train-test Split
    X = data[['concatenation']]
    y = data[['categories']]

    # Train test split --------------------------
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # -------------------------------------------

    # Count Vectorizer & Logistic Regression ----
    vec = CountVectorizer(max_features=5000)
    clf = LogisticRegression()
    pipeline = make_pipeline(vec, clf)
    pipeline.fit(x_train.concatenation, y_train.categories)
    # -------------------------------------------

    #print_evaluation_report(pipeline)
    print_coefficients()
    create_html_visualization()
    misclassified_examples(pipeline)


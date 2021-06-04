from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import LabelPowerset
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from skmultilearn.adapt import MLkNN
from sklearn.ensemble import RandomForestClassifier
import math
import time

# Options for pandas -----
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def LabelPowersetClassification(x_train, x_test, y_train, y_test, classifier):

    print("\nLabel Powerset evaluation:")

    start = time.time()
    # lp_classifier = LabelPowerset(LogisticRegression())
    lp_classifier = LabelPowerset(classifier)
    lp_classifier.fit(x_train, y_train)
    lp_predictions = lp_classifier.predict(x_test)

    print("Accuracy = ", accuracy_score(y_test, lp_predictions))
    print("F1 score = ", f1_score(y_test, lp_predictions, average="micro"))
    print("F1 score = ", f1_score(y_test, lp_predictions, average="macro"))
    print("F1 score = ", f1_score(y_test, lp_predictions, average="wighted"))
    print("F1 score = ", f1_score(y_test, lp_predictions, average="sample"))
    print("Hamming loss = ", hamming_loss(y_test, lp_predictions))

    stop = time.time()
    print("The time of the LabelPowersetClassification run:", stop - start)


def BinaryRelevanceClassification(x_train, x_test, y_train, y_test, classifier):

    print("\nBinary Relevance evaluation:")

    start = time.time()
    # br_classifier = BinaryRelevance(GaussianNB())
    br_classifier = BinaryRelevance(classifier)
    br_classifier.fit(x_train, y_train)
    br_predictions = br_classifier.predict(x_test)

    print("Accuracy = ", accuracy_score(y_test, br_predictions.toarray()))
    print("F1 score = ", f1_score(y_test, br_predictions, average="micro"))
    print("F1 score = ", f1_score(y_test, br_predictions, average="macro"))
    print("F1 score = ", f1_score(y_test, br_predictions, average="wighted"))
    print("F1 score = ", f1_score(y_test, br_predictions, average="sample"))
    print("Hamming loss = ", hamming_loss(y_test, br_predictions))

    stop = time.time()
    print("The time of the LabelPowersetClassification run:", stop - start)


def ClassifierChainsClassification(x_train, x_test, y_train, y_test, classifier):

    print("\nClassifier Chains evaluation:")

    start = time.time()
    # classifier = ClassifierChain(LogisticRegression())
    classifier = ClassifierChain(classifier)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    print("Accuracy = ", accuracy_score(y_test, predictions.toarray()))
    print("F1 score = ", f1_score(y_test, predictions, average="micro"))
    print("F1 score = ", f1_score(y_test, predictions, average="macro"))
    print("F1 score = ", f1_score(y_test, predictions, average="wighted"))
    print("F1 score = ", f1_score(y_test, predictions, average="sample"))
    print("Hamming loss = ", hamming_loss(y_test, predictions))

    stop = time.time()
    print("The time of the LabelPowersetClassification run:", stop - start)


def MLkNNClassification(x_train, x_test, y_train, y_test, nearest_k):

    print("\nMLkNN evaluation:")

    start = time.time()
    classifier = MLkNN(k=nearest_k)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    print("Accuracy = ", accuracy_score(y_test, predictions.toarray()))
    print("F1 score = ", f1_score(y_test, predictions, average="micro"))
    print("Hamming loss = ", hamming_loss(y_test, predictions))

    stop = time.time()
    print("The time of the MLkNN k=10 run:", stop - start)


if __name__ == '__main__':
    df = pd.read_csv("../data/category_columns_dataset.csv",
                     sep=',',
                     header=0,
                     skiprows=0)
    # df = df.sample(n=1000, replace=True, random_state=42)
    print(df.head(5))

    # train test split
    train, test = train_test_split(df, random_state=42, test_size=0.30, shuffle=True)

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

    # print(x_train)
    # drop all other labels apart from target
    y_train = train.drop(labels=['concatenation',
                                 'categories',
                                 'abstract',
                                 'title',
                                 'Unnamed: 0',
                                 'Unnamed: 0.1'
                                 ],
                         axis=1)
    y_test = test.drop(labels=['concatenation',
                               'categories',
                               'abstract',
                               'title',
                               'Unnamed: 0',
                               'Unnamed: 0.1'
                               ],
                       axis=1)

    # transform Y label to array
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
 
    # print tf-idf scores
    print('tf_idf scores: \n', sorted(list(zip(vectorizer.get_feature_names(),
                                               x_train.sum(0).getA1())),
                                      key=lambda x: x[1], reverse=True)[:40])

    # call Multi-label Classification methods
    print(x_train)
    print(x_test)
    print(len(y_train))
    print(len(y_test))

    k = round(math.sqrt(len(y_train)))

    classifiers = [LogisticRegression(), GaussianNB(), RandomForestClassifier()]

    for classifier in classifiers:
        LabelPowersetClassification(x_train, x_test, y_train, y_test, classifier)
        BinaryRelevanceClassification(x_train, x_test, y_train, y_test, classifier)
        ClassifierChainsClassification(x_train, x_test, y_train, y_test, classifier)
    MLkNNClassification(x_train, x_test, y_train, y_test, k)


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


def LabelPowersetClassification():
    lp_classifier = LabelPowerset(LogisticRegression())
    lp_classifier.fit(x_train, y_train)
    lp_predictions = lp_classifier.predict(x_test)

    print("\nLabel Powerset evaluation:")
    print("Accuracy = ", accuracy_score(y_test, lp_predictions))
    print("F1 score = ", f1_score(y_test, lp_predictions, average="micro"))
    print("Hamming loss = ", hamming_loss(y_test, lp_predictions))


def BinaryRelevanceClassification():
    br_classifier = BinaryRelevance(GaussianNB())
    br_classifier.fit(x_train, y_train)
    br_predictions = br_classifier.predict(x_test)

    print("\nBinary Relevance evaluation:")
    print("Accuracy = ", accuracy_score(y_test, br_predictions.toarray()))
    print("F1 score = ", f1_score(y_test, br_predictions, average="micro"))
    print("Hamming loss = ", hamming_loss(y_test, br_predictions))


def ClassifierChainsClassification():
    classifier = ClassifierChain(LogisticRegression())
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    print("\nClassifier Chains evaluation:")
    print("Accuracy = ", accuracy_score(y_test, predictions.toarray()))
    print("F1 score = ", f1_score(y_test, predictions, average="micro"))
    print("Hamming loss = ", hamming_loss(y_test, predictions))


if __name__ == '__main__':

    # read data
    df = pd.read_csv("../data/category_columns_dataset.csv",
                     sep=',',
                     header=0,
                     skiprows=0).head(400)

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

    y_train = train.drop(labels=['abstract', 'title', 'concatenation', 'categories', 'Unnamed: 0', 'Unnamed: 0.1'],
                         axis=1)
    y_test = test.drop(labels=['abstract', 'title', 'concatenation', 'categories', 'Unnamed: 0', 'Unnamed: 0.1'],
                       axis=1)

    print('tf_idf scores: \n', sorted(list(zip(vectorizer.get_feature_names(),
                                               x_train.sum(0).getA1())),
                                      key=lambda x: x[1], reverse=True)[:40])


    LabelPowersetClassification()
    BinaryRelevanceClassification()
    #ClassifierChainsClassification()


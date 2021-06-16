from alibi.explainers import AnchorText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import spacy
from alibi.utils.download import spacy_model


if __name__ == '__main__':
    data = pd.read_csv("../data/multi_class/dataset_multiclass_preprocessed_big.csv")
    #print(len(data))

    X = data[['concatenation']]
    y = data[['categories']]


    # Train-test Split
    # ----------------------------------------------------------
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # ----------------------------------------------------------


    # CountVectorizer & Fitting the classifier
    vec = CountVectorizer(max_features=5000)
    # ----------------------------------------------------------
    clf = LogisticRegression() #LogisticRegressionCV()
    pipeline = make_pipeline(vec, clf)
    pipeline.fit(x_train.concatenation, y_train.categories)
    # ----------------------------------------------------------


    # Anchors
    # ----------------------------------------------------------
    model = 'en_core_web_md'
    spacy_model(model=model)
    nlp = spacy.load(model)

    INDEX = 150
    predict_fn = lambda x: clf.predict(vec.transform(x))
    explainer = AnchorText(nlp, predict_fn)
    explanation = explainer.explain(x_train['concatenation'].values[INDEX], threshold=0.95, use_similarity_proba=False,
                                    use_unk=True, sample_proba=0.5)

    # max_pred = 2
    print(explanation)
    print('Key Singal from Anchors: %s' % (' AND '.join(explanation.anchor)))
    print('Precision: %.2f' % explanation.precision)
    # ----------------------------------------------------------


    #----------------------------------------------------------
    class_names = data["categories"].explode().unique()
    text = x_train['concatenation'].values[INDEX]
    print(text)

    pred = predict_fn([text])[0]

    print('Prediction: %s' % pred)
    print('Anchor: %s' % (' AND '.join(explanation.anchor)))
    print('Precision: %.2f' % explanation.precision)
    print('\nExamples where anchor applies and model predicts %s:' % pred)
    print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_true']]))

    #alternative = class_names[1 - predict_fn([text])[0]]
    #print('\nExamples where anchor applies and model predicts %s:' % alternative)
    #print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_false']]))
    # ----------------------------------------------------------





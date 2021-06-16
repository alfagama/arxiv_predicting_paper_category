from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


if __name__ == '__main__':
    data = pd.read_csv("../data/multi_class/dataset_multiclass_preprocessed.csv")#.head(3000)
    #print(data)
    
    data = shuffle(data, random_state=22)
    data['class_label'] = data['categories'].factorize()[0] # convert label to numeric

    text = data["concatenation"].tolist()
    labels = data["class_label"].tolist()

    # Train test split --------------------------
    x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=40)
    # -------------------------------------------
        
    # Count Vectorizer --------------------------
    vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), stop_words='english',
                                 binary=True, max_features=5000)
    train_vectors = vectorizer.fit_transform(x_train)
    test_vectors = vectorizer.transform(x_test)
    # -------------------------------------------

    # Fit the model --------------------------
    model = LogisticRegression(n_jobs=1, C=1e5)
    model.fit(train_vectors, y_train)
    pred = model.predict(test_vectors)
    # -------------------------------------------

    # Model evaluation -----------------------
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='weighted')
    recall = recall_score(y_test, pred, average='weighted')
    f1 = f1_score(y_test, pred, average='weighted')
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f\n" % (accuracy, precision, recall, f1))
    # -------------------------------------------

    # LIME --------------------------------
    c = make_pipeline(vectorizer, model)
    class_names = list(data.categories.unique())
    explainer = LimeTextExplainer(class_names=class_names)

    idx = 500
    exp = explainer.explain_instance(x_test[idx], c.predict_proba, num_features=6, labels=[0,1,2,3])
    print(x_test[idx])
    print('Document id: %d' % idx)
    print('Predicted class:', class_names[model.predict(test_vectors[idx]).reshape(1, -1)[0, 0]])
    print('True class: %s' % class_names[y_test[idx]])

    category_id = model.predict(test_vectors[idx]).reshape(1, -1)[0, 0]
    print('\nCategory id: %d' % model.predict(test_vectors[idx]).reshape(1, -1)[0, 0])
    print('Explanation for class %s' % class_names[1])
    print('\n'.join(map(str, exp.as_list(label=category_id))))

    #exp = explainer.explain_instance(x_test[idx], c.predict_proba, num_features=6, top_labels=2)
    #print(exp.available_labels())

    exp.save_to_file("lime2.html")
    # -------------------------------------------


# ArXiv - Predicting Scientific Paper Category
----------------------------------------------------
## MSc Data & Web Science, Aristotle University of Thessaloniki (AUTH)
### Course: Advanced Machine Learning
#### Project: *“Predicting Categories of Scientific Papers with Advanced Machine Learning Techniques”*

----------------------------------------------------
**Team Members**:
1. Georgios Arampatzis
2. Alexia Fytili
3. Eleni Tsiolaki

----------------------------------------------------
**Dataset**:
[ArXiv](https://www.kaggle.com/Cornell-University/arxiv)

----------------------------------------------------
**Imbalance**:

*Oversampling*: SMOTE, Borderline SMOTE, RandomOverSampler

*Undersampling*: Tomek Links, Random Undersampling, NearMiss1, NearMiss2, NearMiss3

*Over&Undersampling*: SMOTE and Tomek Links

*Extra*: Generating Synonym Text

----------------------------------------------------
**Multi-label Classification**:
- LabelPowerset
- BinaryRelevance
- ClassifierChain
- ML*k*NN

----------------------------------------------------
**Interpretability / Explainability**:
- ELI5
- LIME
- Anchors
- SHAP

----------------------------------------------------
**Code Structure**:
```
.
└── arxiv_predicting_paper_category
    ├── Imbalanced
    │   ├── imbalance_generate_synonym_text.py
    │   ├── imbalance_new.py
    │   ├── imbalanced_last.py
    │   └── imbalanced_methods.py
    ├── Interpretability
    │   ├── LIME.py
    │   └── eli5.py
    ├── Multi_label_classification
    │   ├── multi_label_classification_new.py
    │   ├── multi_label_methods.py
    │   └── multi_label_plots.py
    ├── data_pics
    │   ├── imbalanced_cv.png
    │   ├── imbalanced_dataset.png
    │   └── imbalanced_roc.png
    ├── dataset
    │   ├── create_synonym_dataset.py
    │   ├── dataset_methods.py
    │   └── exploratory_data_analysis.py
    ├── preprocessing
    │   ├── create_csv_with_categories_as_new_Columns.py
    │   ├── create_multi_class_dataset.py
    │   ├── create_preprocessed_csv.py
    │   ├── create_tf_idf_csv.py
    │   ├── dictionaries.py
    │   ├── filter_dataset_based_on_category.py
    │   └── main_preprocessing.py
    ├── .gitingnore
    └── README.md
```

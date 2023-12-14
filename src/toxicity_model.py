from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.textprep import preprocess_data_frame


def build_toxicity_model(*, max_features: int, random_state: int) -> Pipeline:
    return Pipeline([
        ('preprocess', FunctionTransformer(preprocess_data_frame)),
        ('tfidf', TfidfVectorizer(use_idf=True, max_features=max_features)),
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=6))),
    ], memory='../tmp')

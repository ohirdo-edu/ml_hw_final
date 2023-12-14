import numpy as np
import pandas as pd

from src.helpers import read_comments_toxicity_csv, stratified_train_test_split, predict_toxicity_probs, LABEL_COLUMNS


def test_read_comments_toxicity_csv():
    x, y = read_comments_toxicity_csv("tests/sample_data.csv")
    assert len(x) == 2
    assert len(y) == 2
    assert x.columns.tolist() == ["comment_text"]
    assert y.columns.tolist() == LABEL_COLUMNS


def test_stratified_train_test_split():
    x = pd.DataFrame({"feature": list("abcdefghi")})
    y = pd.DataFrame({"label": [1, 1, 1, 1, 2, 2, 2, 2, 3]})
    x_train, x_test, y_train, y_test = stratified_train_test_split(x, y, random_state=42)
    assert 1 in y_train['label'].tolist()
    assert 2 in y_train['label'].tolist()
    assert 1 in y_test['label'].tolist()
    assert 2 in y_test['label'].tolist()


class ModelMock:
    def __init__(self, probs):
        self.probs = probs

    def predict_proba(self, x):
        return self.probs


def test_predict_toxicity_probs():
    x = pd.DataFrame({"feature": ['a', 'b']})
    probs = np.array([
        [[0.2, 0.8], [0.3, 0.7]],
        [[0.2, 0.8], [0.3, 0.7]],
        [[0.2, 0.8], [0.3, 0.7]],
        [[0.2, 0.8], [0.3, 0.7]],
        [[0.2, 0.8], [0.3, 0.7]],
        [[0.2, 0.8], [0.3, 0.7]],
    ])

    flat_probs = predict_toxicity_probs(x, ModelMock(probs))

    assert np.array_equal(flat_probs.values, np.array([
        [0.8] * 6,
        [0.7] * 6,
    ]))

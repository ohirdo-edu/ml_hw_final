import pandas as pd
from sklearn.model_selection import train_test_split
from os import PathLike


LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def read_comments_toxicity_csv(path: PathLike | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_full = pd.read_csv(path, index_col="id")
    return df_full[['comment_text']], df_full[LABEL_COLUMNS]


def stratified_train_test_split(
        x, y, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from collections import defaultdict
    tuple_to_id = defaultdict(list)
    for index, row in y.iterrows():
        tuple_to_id[tuple(row.values)].append(index)
    indexes_to_drop = [indexes[0] for t, indexes in tuple_to_id.items() if len(indexes) == 1]

    x_reduced = x.drop(indexes_to_drop)
    y_reduced = y.drop(indexes_to_drop)

    x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(
        x_reduced, y_reduced, stratify=y_reduced, random_state=random_state
    )

    return (
        pd.concat([x_train_raw, x.loc[indexes_to_drop]]),
        x_test_raw,
        pd.concat([y_train_raw, y.loc[indexes_to_drop]]),
        y_test_raw,
    )


def predict_toxicity_probs(x: pd.DataFrame, model):
    res = pd.DataFrame(index=x.index)
    y_pred = model.predict_proba(x)
    for label_index, label in enumerate(LABEL_COLUMNS):
        res[label] = y_pred[label_index][:, 1]

    return res

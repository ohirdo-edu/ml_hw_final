from pathlib import Path
import joblib

import click
from sklearn.metrics import roc_auc_score

from .helpers import read_comments_toxicity_csv, stratified_train_test_split, predict_toxicity_probs
from .toxicity_model import build_toxicity_model


@click.command()
@click.option("--max_features", type=int, help="Max features")
@click.option("--random_state", default=42, type=int, help="Random state")
@click.option("--output_folder", default="checkpoints", type=str, help="Output folder")
def main(max_features, random_state, output_folder):
    model = build_toxicity_model(max_features=max_features, random_state=random_state)
    x_full, y_full = read_comments_toxicity_csv('data/train.csv')
    x_train, x_test, y_train, y_test = stratified_train_test_split(x_full, y_full, random_state=random_state)
    model.fit(x_train, y_train)

    print('Training score:', roc_auc_score(y_train, predict_toxicity_probs(x_train, model)))
    print('Test score:', roc_auc_score(y_test, predict_toxicity_probs(x_test, model)))

    model.fit(x_full, y_full)

    output_directory = Path(output_folder)
    output_directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_directory / f"model_{max_features}.joblib")


if __name__ == '__main__':
    main()

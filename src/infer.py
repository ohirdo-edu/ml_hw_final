import click
import joblib
import pandas as pd

from .helpers import predict_toxicity_probs


@click.command()
@click.option('--model_path', required=True, type=click.Path(exists=True))
@click.option('--input_path', required=True, type=click.Path(exists=True))
@click.option('--output_path', required=True, type=click.Path())
def main(model_path, input_path, output_path):
    model = joblib.load(model_path)
    x = pd.read_csv(input_path, index_col='id')[['comment_text']]
    result_df = predict_toxicity_probs(x, model)
    result_df.to_csv(output_path, index=True)


if __name__ == '__main__':
    main()

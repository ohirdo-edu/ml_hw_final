import click
import joblib
import pandas as pd
import requests
import io

from .helpers import predict_toxicity_probs


@click.command()
@click.option('--model_url', required=True, type=str, help='URL to the trained model')
def main(model_url):
    r = requests.get(model_url, stream=True)
    r.raise_for_status()
    model = joblib.load(io.BytesIO(r.content))
    while True:
        comment = input()
        x = pd.DataFrame([[comment]], columns=['comment_text'])
        probs = predict_toxicity_probs(x, model)
        print(probs)


if __name__ == '__main__':
    main()

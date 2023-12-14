

## Структура:
- [Github action на лінтер на тести](.github/workflows/testing.yaml)
- [Датасет зі змагання](data)
- [Ноутбук з розглядом датасету](EDA/hw4.ipynb)
- [Допоміжний код, CLI скрипти](src)
- [Тести](tests)
- [Makefile з типовими командами](Makefile)

## Типові задачі

- натренувати модель із заданим гіперпараметром, із [збереженням моделі](checkpoints)

  `python -m src.train --max_features=10001`

- провести інференс для змагання

  `python -m src.infer --model_path checkpoints/model_10001.joblib --input_path data/test.csv --output_path tmp/my_submission.csv`

- запустити інтерактивний інференс з викачуванням моделі за URL

  `python -m src.infer_online --model_url 'https://github.com/ohirdo-edu/ml_hw_final/raw/main/checkpoints/model_10001.joblib'`

- запустити тести

  `make test`

- запустити тести з coverage

  `make coverage`

- запустити тести з coverage

  `make lint`

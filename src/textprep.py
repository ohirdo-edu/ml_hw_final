import nltk
import re
from nltk.tokenize.toktok import ToktokTokenizer
import pandas as pd


nltk.download('stopwords', quiet=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')


def remove_special_characters(text: str):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


def stem_sentence(text: str):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split() if len(word) < 100])
    return text


def remove_stopwords(text: str, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def tokenize_sentence(s: str):
    assert isinstance(s, str)
    s = remove_special_characters(s)
    s = stem_sentence(s)
    s = remove_stopwords(s)
    return s


def preprocess_data_frame(x: pd.DataFrame):
    return x.apply(lambda s: s.transform(tokenize_sentence))['comment_text']

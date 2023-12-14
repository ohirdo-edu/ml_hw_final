from src.textprep import remove_special_characters, tokenize_sentence


def test_remove_special_characters():
    assert remove_special_characters('') == ''
    assert remove_special_characters(r'a bc&^%1 23') == 'a bc1 23'


def test_tokenize_sentence():
    assert tokenize_sentence('') == ''
    sentence = '\nI must admit that I had never heard of Ms. Stewart until she became famous for her legal problems.'
    expected_tokens = 'must admit never heard ms stewart becam famou legal problem'
    assert tokenize_sentence(sentence) == expected_tokens

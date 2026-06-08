""" adapted from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and
eval time. Select them via the "text_cleaners" hyperparameter. For Bulgarian use
"bulgarian_cleaners"; the heavy normalisation (number expansion, foreign-char
dropping, repeat collapsing) lives in text/bulgarian.py.
'''

import re

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def bulgarian_cleaners(text):
    '''Pipeline for Bulgarian. Only lowercases and collapses whitespace; it must
    NOT transliterate to ASCII (that would destroy Cyrillic). Heavy text
    normalisation is handled upstream in text/bulgarian.py.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

from torchtext.datasets import AmazonReviewPolarity
from torchtext.vocab import vocab,build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

from collections import OrderedDict

import constants as CONSTANTS

def _get_data_itr(split):
    data_itr = AmazonReviewPolarity(split=split)
    
    data_itr = iter(map(lambda x:(x[0]-1,x[1]),data_itr))
    return data_itr

def _get_tokenizer():
    
    return get_tokenizer("basic_english")


def build_vocab(data_itr,tokenizer):
    
    
    get_tokenized_text = lambda x:tokenizer(x[1])
    
    v = build_vocab_from_iterator(map(get_tokenized_text,data_itr),min_freq=CONSTANTS.min_freq,specials=["<unk>"])
    v.set_default_index(v["<unk>"])
    
    
    return v
# Copyright (c) Microsoft Corporation. All rights reserved.

# This script reuses some code from
# https://github.com/huggingface/transformers/blob/master/examples/run_glue.py

"""Tokenizer class"""

import sentencepiece as spm
from pytorch_pretrained_bert.tokenization import BertTokenizer
from data_utils.gpt2_bpe import get_encoder
from experiments.exp_def import EncoderModelType
from .utils import load_dict, _truncate_seq_pair

MAX_SEQ_LEN = 512

class Language(str, Enum):
    """An enumeration of the supported pretrained models and languages."""

    ENGLISH: str = "bert-base-uncased"
    ENGLISHCASED: str = "bert-base-cased"
    ENGLISHLARGE: str = "bert-large-uncased"
    ENGLISHLARGECASED: str = "bert-large-cased"
    ENGLISHLARGEWWM: str = "bert-large-uncased-whole-word-masking"
    ENGLISHLARGECASEDWWM: str = "bert-large-cased-whole-word-masking"
    CHINESE: str = "bert-base-chinese"
    MULTILINGUAL: str = "bert-base-multilingual-cased"

class RoBERTaTokenizer(object):
    def __init__(self, vocab, encoder):
        self.vocab = vocab
        self.encoder = encoder

    def encode(self, text):
        ids = self.encoder.encode(text)
        ids = list(map(str, ids))
        if len(ids) > MAX_SEQ_LEN - 2:
            ids = ids[: MAX_SEQ_LEN - 2]
        ids = [0] + [self.vocab[w] if w in self.vocab else self.vocab['<unk>']
                     for w in ids] + [2]
        return ids

    def encode_pair(self, text1, text2):
        ids1 = self.encoder.encode(text1)
        ids1 = list(map(str, ids1))
        ids1 = [self.vocab[w] if w in self.vocab else self.vocab['<unk>']
                for w in ids1] + [2]

        ids2 = self.encoder.encode(text2)
        ids2 = list(map(str, ids2))
        ids2 = [self.vocab[w] if w in self.vocab else self.vocab['<unk>']
                for w in ids2] + [2]
        _truncate_seq_pair(ids1, ids2, MAX_SEQ_LEN -2)
        ids = [0] + ids1 + [2] + ids2
        return ids

class Tokenizer:
    def __init__(self, 
                 language=Language.ENGLISH, 
                 to_lower=False, 
                 encoder_model=EncoderModelType.ROBERTA,
                 roberta_path=None,
                 cache_dir="."，
                 **kwargs):
        """Initializes the underlying pretrained tokenizer for MT-DNN.

        Args:
            language (Language, optional): The pretrained model's language.
                                           Defaults to Language.ENGLISH.
            to_lower (bool, optional): Whether to lower case the input.

            cache_dir (str, optional): Location of BERT's cache directory.
                Defaults to ".".
        """
        if encoder_model == EncoderModelType.ROBERTA:
            if roberta_path is None or (
                    not os.path.exists(roberta_path)):
                print("Please specify roberta model path")
            encoder = get_encoder("{}/encoder.json".format(roberta_path),
                                "{}/vocab.bpe".format(roberta_path))
            vocab = load_dict("{}/ict.txt".format(roberta_path))
            self.tokenizer = RoBERTaTokenizer(vocab, encoder)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                language, do_lower_case=to_lower, cache_dir=cache_dir, **kwargs)

        self.language = language

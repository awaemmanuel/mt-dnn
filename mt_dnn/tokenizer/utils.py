# Copyright (c) Microsoft Corporation. All rights reserved.

from data_utils.vocab import Vocabulary

def load_dict(path):
    vocab = Vocabulary(neat=True)
    # ROBERTA specific tokens
    # "<s>", "<pad>", "</s>", "<unk>"
    vocab.add("<s>")
    vocab.add("<pad>")
    vocab.add("</s>")
    vocab.add("<unk>")
    with open(path, "r", encoding="utf8") as reader:
        for line in reader:
            idx = line.rfind(" ")
            if idx == -1:
                raise ValueError(
                    "Incorrect dictionary format, expected "<token> <cnt>"")
            word = line[:idx]
            vocab.add(word)
    return vocab

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copyed from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that"s truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
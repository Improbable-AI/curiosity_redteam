from typing import List, Callable, Union
import os
import random
import numpy as np
from multiprocessing import Pool

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from fast_bleu import BLEU
    
class SelfBleuReward(object):

    def __init__(self, 
                 grams: List[int] = [3, 4, 6], 
                 sample_size: int = -1,
                 tokenizer: Callable = nltk.word_tokenize,) -> None:
        print("BLEU sample size: ", sample_size)
        self.references = []
        self.grams = grams
        self.sample_size = sample_size
        self.tokenizer = tokenizer

    def append_reference(self, ref: Union[str, List[str]]):
        if isinstance(ref, list):
            self.references += list(map(self.tokenizer, ref))
        else:
            self.references.append(self.tokenizer(ref))

    def __call__(self, hypotheses: List[str]):
        weights = {f"{n}-gram": ([1. / n] * n) for n in self.grams}
        if self.sample_size > 0:
            sample_size = min(len(self.references), self.sample_size)
            bleu = BLEU(random.sample(self.references, k=sample_size), weights)
        else:
            bleu = BLEU(self.references, weights)
        tokenized_hypotheses = list(map(self.tokenizer, hypotheses))
        scores = list(bleu.get_score(tokenized_hypotheses).values())
        return np.asarray(scores).mean(axis=0)

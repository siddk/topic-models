"""
hdp.py

Core class containing implementation of the Hierarchical Dirichlet Process - Latent Dirichlet
Allocation (HDP - LDA) topic model from scratch, using the Chinese Restaurant Process and Collapsed
Gibbs Sampling.

Implementation derived from Infinite LDA - Implementing the HDP with minimum code complexity by
Gregor Heinrich and Hierarchical Dirichlet Processes (Teh et. al) implemented by Nakatani Shuyo:

http://www.arbylon.net/publications/ilda.pdf
http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf
https://github.com/shuyo/iir/blob/master/lda/hdplda.py
"""
from collections import defaultdict
import numpy as np


class HDP:
    def __init__(self, corpus, alpha=1.0, beta=0.1, gamma=1.5, k=1, iterations=100):
        """
        Instantiate an HDP-LDA Topic Model with the given corpus.

        :param corpus: Corpus object containing documents and vocabulary for the given model.
        :param alpha: Dirichlet Hyperparameter, determines prior for document distribution,
                      alpha = 1 implies a symmetric prior.
        :param beta: Dirichlet Hyperparameter, determines prior for word distributions given topics.
        :param gamma: Dirichlet Hyperparameter, determines Chinese Restaurant Prior for determining
                      number of topics.
        :param k: Optional Number of Topics (default = 1).
        :param iterations: Number of iterations of gibbs sampling to perform.
        """
        self.corpus, self.docs, self.V = corpus, corpus.docs, len(corpus.vocab)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma


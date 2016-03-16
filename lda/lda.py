"""
lda.py

Core class containing implementation of the Latent Dirichlet Allocation topic model from
scratch, using Collapsed Gibbs Sampling.
"""

from corpus.corpus import test_dict, Corpus


class LDA:
    def __init__(self, corpus, k, iterations, alpha=1.0, eta=.001):
        """
        Instantiate an LDA Topic Model with a given Corpus.

        :param corpus: Corpus object containing documents and vocabulary for the given model.
        :param k: Number of topics (clusters) to separate data into.
        :param iterations: Number of sampling iterations for inference.
        :param alpha: Dirichlet Hyperparameter, determines prior topic cluster probability (alpha=1)
                      implies a symmetric prior.
        :param eta: Learning rate?
        """
        self.corpus = corpus
        self.k, self.iterations, self.alpha, self.eta = k, iterations, alpha, eta
"""
lda.py

Core class containing implementation of the Latent Dirichlet Allocation topic model from
scratch, using Collapsed Gibbs Sampling.

Implementation derived from Probabilistic Topic Models by Steyvers and Griffiths:
http://psiexp.ss.uci.edu/research/papers/SteyversGriffithsLSABookFormatted.pdf
"""
from collections import defaultdict
from corpus.corpus import test_dict, Corpus
import numpy as np


class LDA:
    def __init__(self, corpus, k, iterations=100, alpha=1.0, beta=.001):
        """
        Instantiate an LDA Topic Model with a given Corpus.

        :param corpus: Corpus object containing documents and vocabulary for the given model.
        :param k: Number of topics (clusters) to separate data into.
        :param iterations: Number of sampling iterations for inference.
        :param alpha: Dirichlet Hyperparameter, determines prior for document distribution,
                      alpha = 1 implies a symmetric prior.
        :param beta: Dirichlet Hyperparameter, determines prior for word distributions given topics.
        """
        self.corpus = corpus
        self.k, self.iterations, self.alpha, self.beta = k, iterations, alpha, beta

        # Count of number of times a word is assigned to a given topic -> indexed wt[topic][word_id]
        self.wt = defaultdict(lambda: defaultdict(float))

        # Count of number of tokens in document assigned to each topic -> indexed dt[doc_id][topic]
        self.dt = defaultdict(lambda: defaultdict(float))

        # Randomly assign each word in each document to a different topic
        self.assignments = self.random_init()

    def random_init(self):
        """
        Randomly assign each word in the corpus to one of K Topics, sampled from a uniform
        distribution.

        :return: List of Numpy Arrays of topic assignments. Each index of the list corresponds to a
                 different document id.
        """
        assignments = []
        for doc_id in range(len(self.corpus.docs)):
            doc = self.corpus.docs[doc_id]
            # Assign all words to different topics drawn from the uniform distribution over K
            assignments.append(np.random.randint(self.k, size=len(doc)))
            for index in range(len(doc)):
                assigned_topic = assignments[doc_id][index]
                word_id = self.corpus.docs[doc_id][index]

                # Increment count of number of times a word_id is assigned a given topic.
                self.wt[assigned_topic][word_id] += 1

                # Increment count of number of times words in a document are assigned a given topic
                self.dt[doc_id][assigned_topic] += 1

        return assignments

    def collapsed_gibbs(self):
        """
        Run Collapsed Gibbs sampling for iterations. The goal of Collapsed Gibbs Sampling is to
        integrate out over the parameters in questions, phi (the posterior for the probability of
        words given topics), and theta (the posterior for the probability of topics given documents).

        This leaves us with a single macro-posterior probability, p(z_i = j | z_{-i}, w_i, d_i), the
        probability that a word token (i) is assigned to a topic j (z_i = j), given all the other
        word - topic assignments in the corpus (z_{-i}), as well as the current word (w_i), and the
        current document (d_i).

        As we're using Gibbs Sampling (following a Markov Chain Monte Carlo)
        procedure, after enough iterations, we're converging on the actual true posterior
        distribution, which is why our topic assignments get more accurate over time.
        """
        for _ in range(self.iterations):
            for doc_id in range(len(self.corpus.docs)):
                doc = self.corpus.docs[doc_id]
                for index in range(len(doc)):
                    word_id = doc[index]

                    # Remove the current word from the existing word - topic, topic - doc counts.
                    assigned_topic = self.assignments[doc_id][index]
                    self.wt[assigned_topic][word_id] -= 1
                    self.dt[doc_id][assigned_topic] -= 1

                    # Update topic assignment - Sample from Posterior p(z_i = j | z_{-i}, w_i, d_i)

                    # Number of words in document + (Number of Topics * Alpha)
                    theta_denominator = sum(self.dt[doc_id].values()) + self.k * self.alpha

                    # Number of tokens in each topic + (Number of Words * Beta)
                    phi_denominator = np.zeros(self.k)
                    for tid in range(self.k):
                        phi_denominator[tid] = sum(self.wt[tid].values()) + len(
                            self.corpus.vocab) * self.beta

                    # Compute actual posterior
                    p_z = np.zeros(self.k)
                    for tid in range(self.k):
                        p_z[tid] = ((self.wt[tid][word_id] + self.beta) / phi_denominator[tid]) * (
                            (self.dt[doc_id][tid] + self.alpha) / theta_denominator)

                    # Normalize posterior
                    p_z = map(lambda x: x / sum(p_z), p_z)

                    # Get new topic assignment by sampling from p_z
                    new_topic = np.random.choice(self.k, p=p_z)

                    # Increment word - topic, topic - doc counts.
                    self.assignments[doc_id][index] = new_topic
                    self.wt[new_topic][word_id] += 1
                    self.dt[doc_id][new_topic] += 1

    def get_theta(self):
        """
        After conducting Collapsed Gibbs sampling, estimate theta, the probability of topic given
        document for each document in the corpus.

        :return: List of Lists mapping document ID to probability of each topic.
        """
        theta = []
        for doc_id in range(len(self.corpus.docs)):
            theta.append(np.zeros(self.k))
            for tid in range(self.k):
                theta[doc_id][tid] = (self.dt[doc_id][tid] + self.alpha) / (
                    sum(self.dt[doc_id].values()) + self.alpha * self.k)
        return theta

    def get_phi(self):
        """
        After conducting Collapsed Gibbs sampling, estimate phi, the probability of word given
        topic for each word in the vocabulary.

        :return: List of Lists mapping topic ID to probability of each word.
        """
        phi = []
        for tid in range(self.k):
            phi.append(np.zeros(len(self.corpus.vocab)))
            for word_id in self.corpus.vocab_map:
                phi[tid][word_id] = (self.wt[tid][word_id] + self.beta) / (
                    sum(self.wt[tid].values()) + self.beta * len(self.corpus.vocab))
        return phi


if __name__ == "__main__":
    character_corpus = Corpus(test_dict)
    lda = LDA(character_corpus, k=3)
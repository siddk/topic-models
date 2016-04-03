"""
hdp.py

Core class containing implementation of the Hierarchical Dirichlet Process - Latent Dirichlet
Allocation (HDP - LDA) topic model from scratch, using the Chinese Restaurant Process and Collapsed
Gibbs Sampling.

Implementation derived from Infinite LDA - Implementing the HDP with minimum code complexity by
Gregor Heinrich:
http://www.arbylon.net/publications/ilda.pdf
"""
from collections import defaultdict
import numpy as np
import random

# Number of Steps to increase Posterior Sampling Array
PP_STEP = 10


class HDP:
    def __init__(self, corpus, k=0, iterations=100, alpha=1.0, beta=0.1, gamma=1.5):
        """
        Instantiate an HDP-LDA Topic Model with the given corpus.

        :param corpus: Corpus object containing documents and vocabulary for the given model.
        :param k: Optional Number of Topics (can be 0 if gamma > 0).
        :param iterations: Number of iterations of gibbs sampling to perform.
        :param alpha: Dirichlet Hyperparameter, determines prior for document distribution,
                      alpha = 1 implies a symmetric prior.
        :param beta: Dirichlet Hyperparameter, determines prior for word distributions given topics.
        :param gamma: Dirichlet Hyperparameter, determines Chinese Restaurant Prior for determining
                      number of topics (0 for fixed K --> regular LDA).
        """
        self.corpus, self.M, self.V = corpus, len(corpus.docs), len(corpus.vocab)
        self.K, self.fixed_k = k, gamma == 0
        self.alpha, self.beta, self.gamma, self.iterations = alpha, beta, gamma, iterations

        # Count of number of tokens in document assigned to each topic -> indexed dt[doc_id][topic]
        self.dt = defaultdict(lambda: defaultdict(float))

        # Count of number of times a word is assigned to a given topic -> indexed wt[topic][word_id]
        self.wt = defaultdict(lambda: defaultdict(float))

        # Count of overall topic assignments
        self.t = defaultdict(float)

        # Initialize Assignments, where each index is a different document -> z[doc_id][word_index]
        self.z = [np.zeros(len(x)) for x in self.corpus.docs]

        # Initialize lists of Active and Inactive topics
        self.k_active, self.k_gaps = [], []

        # Create Mean Topic Weights --> Root Dirichlet for CRP
        self.tau = []

        # Create Posterior Sampling Array (Pr(z | *))
        self.p_z = []

        # Initialize
        self.initialized = False
        self.initialize()

    def initialize(self):
        """
        Set up Markov Chain by resetting all count dictionaries, setting tau, building k_active,
        k_gaps, initializing Posterior Sampling array.
        """
        for k in range(self.K):
            self.k_active.append(k)
            self.tau.append(1.0 / float(self.K))

        # Tau is defined to be an array of length K + 1 (other tables in CRP).
        try:
            self.tau.append(1.0 / float(self.K))
        except ZeroDivisionError:
            self.tau.append(1.0)  # Value is just used to set likelihood of next component

        # Set up Posterior Sampling Array (increment by PP_STEP)
        self.p_z = np.zeros(self.K + PP_STEP)

        # Initialize by running one iteration of Gibbs Sampling
        self.collapsed_gibbs(1)

    def collapsed_gibbs(self, n_iter):
        """
        Run Collapsed Gibbs sampling for n_iter iterations. The goal of Collapsed Gibbs Sampling
        under the HDP - LDA Model is similar to LDA. We wish to obtain the single macro-posterior
        probability, p(z_i = j | z_{-i}, w_i, d_i, alpha*tau, beta, gamma, K), the probability that
        a word token (i) is assigned to a topic j (z_i = j), given all the other word - topic
        assignments in the corpus (z_{-i}), as well as the current word (w_i), the current document
        (d_i), the and the current set of Dirichlet Hyperparameters.

        What changes with the HDP-LDA is at every iteration, we add new topics using the Chinese
        Restaurant Process, and we remove any empty topics. At the end of the entire sampling
        process, not only will we have the topic distribution, but we will have automatically
        discovered the proper number of topics for the given dataset.

        :param n_iter: Number of Iterations to run sampling for.
        """
        for _ in range(n_iter):
            for doc_id in range(len(self.corpus.docs)):
                doc = self.corpus.docs[doc_id]
                for index in range(len(doc)):
                    word_id = doc[index]

                    # Ambiguous Topics, Removed Topics
                    k, k_old = -1, -1

                    # Only perform the actual gibbs decrementing if fully initialized
                    if self.initialized:
                        k = self.z[doc_id][index]
                        self.dt[doc_id][k] -= 1
                        self.wt[k][word_id] -= 1
                        self.t[k] -= 1
                        k_old = k

                    # Compute Weights of each Topic
                    p_sum = 0.0

                    # Collapsed Posterior Sampling (alpha * tau[k] determines document-topic ratio)
                    for kk in range(self.K):
                        k = self.k_active[kk]
                        self.p_z[kk] = (self.dt[doc_id][k] + self.alpha * self.tau[k]) * (
                            self.wt[k][word_id] + self.beta) / (self.t[k] + self.V * self.beta)
                        p_sum += self.p_z[kk]

                    # Likelihood of New Topic (all Unseen components)
                    if not self.fixed_k:
                        self.p_z[self.K] = self.alpha * self.tau[self.K] / self.V

                    # Get new assignment (the following code block samples from categorical array)
                    u = random.random()
                    p_sum = 0
                    for kk in range(self.K + 1):
                        p_sum += self.p_z[kk]
                        if u <= p_sum:
                            break
                    new_topic = kk

                    # Reassign and reincrement
                    if new_topic < self.K:
                        k = self.k_active[kk]
                        self.z[doc_id][index] = k
                        self.dt[doc_id][k] += 1
                        self.wt[k][word_id] += 1
                        self.t[k] += 1
                    else:
                        assert not self.fixed_k
                        self.z[doc_id][index] = self.spawn_topic(doc_id, word_id)
                        self.update_tau()

    def spawn_topic(self, d, w):
        """
        Adds a topic to the list of active topics either by reusing an inactive topic index (gap),
        or increasing the count arrays.

        :param d: Current document id.
        :param w: Current word id.
        :return: Index of spawned topic
        """
        k = "NONE"
        if len(self.k_gaps) > 0:
            # Reuse gap
            k = self.k_gaps.pop()
        else:
            # Add element to count arrays
            k = self.K
            self.tau.append(0.0)

        self.k_active.append(k)
        self.dt[d][k] = 1
        self.wt[w][k] = 1
        self.t[k] = 1
        self.K += 1

        if len(self.p_z) <= self.K:
            self.p_z = np.zeros(self.K + PP_STEP)

        return k

    def update_tau(self):
        """
        Prune topics and update the tau array, the root Dirichlet Process mixture weights.
        """
        mk = np.zeros(self.K + 1)
        for kk in range(self.K):
            k = self.k_active[kk]
            for m in range(len(self.corpus.docs)):
                pass



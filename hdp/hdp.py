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
from corpus.corpus import test_dict, Corpus, defaultdict
from scipy.special import gammaln
import numpy as np
import sys


class HDP:
    def __init__(self, corpus, alpha=1.0, beta=0.1, gamma=1.5, iterations=30):
        """
        Instantiate an HDP-LDA Topic Model with the given corpus.

        :param corpus: Corpus object containing documents and vocabulary for the given model.
        :param alpha: Dirichlet Hyperparameter, determines prior for document distribution
        :param beta: Dirichlet Hyperparameter, determines prior for word distributions given topics.
        :param gamma: Dirichlet Hyperparameter, determines Chinese Restaurant Prior for tables.
        :param iterations: Number of iterations of gibbs sampling to perform.
        """
        self.corpus, self.docs = corpus, corpus.docs
        self.V, self.M = len(corpus.vocab), len(corpus.docs)  # V = Vocabulary Size, M = # of Docs
        self.alpha, self.beta, self.gamma = alpha, beta, gamma  # Hyperparameters
        self.iterations = iterations  # Number of sampling iterations

        # Keep track of used topics k -> k = 0 implies draw a new topic
        self.used_topics = [0]

        # Keep track of the tables used for each document j -> 0 implies a new table
        self.used_tables = [[0] for j in xrange(self.M)]

        # Chinese Restaurant Franchise Counts

        # Table Counts (all following indexed doc_id -> table_id)
        self.dk = [np.zeros(1, dtype=int) for j in xrange(self.M)]  # Topics for each document
        self.dt = [np.zeros(1, dtype=int) for j in xrange(self.M)]  # Document-Table Count
        self.dtw = [[None] for j in xrange(self.M)]  # Document-Table-Word Count

        # Dish (Topic) Counts
        self.m = 0
        self.mk = np.ones(1, dtype=int)  # Number of tables for each topic
        self.nk = np.array([self.beta * self.V])  # Number of terms for each topic (plus smoothing)
        self.wt = [defaultdict(0)]  # Word-Topic Count

        # Table for each document and word (-1 means not assigned)
        self.tables = [np.zeros(len(doc), dtype=int) - 1 for doc in self.docs]

        # Perform inference
        self.inference()

    def inference(self):
        """
        Performs the HDP inference -> self.iterations number of Gibbs Sampling Iterations.
        """
        for _ in xrange(self.iterations):
            self.sample()

    def sample(self):
        """
        Performs one iteration of Gibbs Sampling. Each iteration consists of sampling from
        posterior over tables t, then sampling from posterior over topics k.
        """
        # Sample new table for each word
        for doc_id in xrange(self.M):
            doc = self.docs[doc_id]
            for word_id in range(len(doc)):
                self.sample_table(doc_id, word_id)

        # Sample new dish for each table
        for doc_id in xrange(self.M):
            for table in self.used_tables[doc_id]:
                if table != 0:
                    self.sample_dish(doc_id, table)

    def sample_table(self, j, i):
        """
        Samples current table for word i in document j from posterior Dirichlet.

        :param j: Current Document Index
        :param i: Current Word Index
        """
        self.leave_table(j, i)  # Remove current observation from Gibbs counts

        # Get word distribution from counts
        word = self.docs[j][i]
        wt_dist = self.get_wt_distribution(word)
        assert wt_dist[0] == 0  # Sanity check, topic 0 implies draw new topic, shouldn't have value

        # Sample from table posterior
        t_posterior = self.table_posterior(j, wt_dist)

        # Pick new table by sampling from posterior
        sampled_table = self.used_tables[j][np.random.multinomial(1, t_posterior).argmax()]

        if sampled_table == 0:  # Draw brand new table
            # New Table doesn't have a dish (topic), so draw new topic from topic posterior
            d_posterior = self.word_dish_posterior(wt_dist)
            sampled_dish = self.used_topics[np.random.multinomial(1, d_posterior).argmax()]
            if sampled_dish == 0:  # Draw brand new dish (topic)
                sampled_dish = self.new_dish()
            sampled_table = self.new_table(j, sampled_dish)

        self.seat_table(j, i, sampled_table)  # Add back to Gibbs counts

    def sample_dish(self, j, t):
        """
        Samples dish (topic) for current table t in document j.

        :param j: Current Document Index
        :param t: Current Table
        """
        self.leave_dish(j, t)  # Remove current observation from Gibbs counts

        # Sample new dish (topic) from posterior over dishes (topics)
        d_posterior = self.table_dish_posterior(j, t)

        # Pick new dish by sampling from posterior
        sampled_dish = self.used_topics[np.random.multinomial(1, d_posterior).argmax()]

        if sampled_dish == 0:  # Draw brand new dish (topic)
            sampled_dish = self.new_dish()

        self.seat_dish(j, t, sampled_dish)  # Add back to Gibbs counts

    def leave_table(self, j, i):
        """
        Remove current observation from Gibbs sampling counts (table-level).

        :param j: Current Document Index
        :param i: Current Word Index
        """
        curr_table = self.tables[j][i]

        # Ensure table exists
        if curr_table > 0:
            curr_topic = self.dk[j][curr_table]  # Current Dish (Topic)
            assert curr_topic > 0  # Ensure Dish (Topic) exists

            # Decrement
            word = self.docs[j][i]
            self.wt[curr_topic][word] -= 1
            self.nk[curr_topic] -= 1
            self.dt[j][curr_table] -= 1
            self.dtw[j][curr_table][word] -= 1

            # If no more people at table t in current document j, get rid of table:
            if self.dt[j][curr_table] == 0:
                self.remove_table(j, curr_table)

    def seat_table(self, j, i, new_table):
        """
        Seat observation at its newly sampled table.

        :param j: Current Document Index
        :param i: Current Word Index
        :param new_table: New Table assignment
        """
        assert new_table in self.used_tables[j]  # Sanity check

        # Update table counts
        self.tables[j][i] = new_table
        self.dt[j][new_table] += 1

        # Update dish (topic) counts
        new_dish = self.dk[j][new_table]
        self.nk[new_dish] += 1

        # Update word counts
        word = self.docs[j][i]
        self.wt[new_dish][word] += 1
        self.dtw[j][new_table][word] += 1

    def leave_dish(self, j, t):
        """
        Removes current dish placement from Gibbs sampling counts.

        :param j: Current Document Index
        :param t: Current Table
        """
        dish = self.dk[j][t]
        assert dish > 0  # Sanity check, checks dish (topic) exists
        assert self.mk[dish] > 0  # Sanity check, assert total topic count is a positive integer

        # Decrement counts
        self.mk[dish] -= 1
        self.m -= 1

        # If count = 0, indicate a new dish (topic) needs to be drawn
        if self.mk[dish] == 0:
            self.used_topics.remove(dish)
            self.dk[j][t] = 0

    def seat_dish(self, j, t, new_dish):
        """
        Serve newly sampled dish at table.

        :param j: Current Document Index
        :param t: Current Table Index
        :param new_dish: Newly sampled dish (topic)
        """
        self.m += 1
        self.mk[new_dish] += 1

        old_dish = self.dk[j][t]
        if old_dish != new_dish:
            self.dk[j][t] = new_dish
            t_count = self.dt[j][t]

            if old_dish != 0:
                self.nk[old_dish] -= t_count

            self.nk[new_dish] += t_count
            for word, count in self.dtw[j][t].iteritems():
                if old_dish != 0:
                    self.wt[old_dish][word] -= count
                self.wt[new_dish][word] += count

    def remove_table(self, j, t):
        """
        Remove table t from restaurant j (document j).

        :param j: Current Document Index
        :param t: Table Index in Current Document.
        """
        curr_topic = self.dk[j][t]  # Current Table Dish (Topic)
        self.used_tables[j].remove(t)  # Remove table from document j's set of used tables

        self.mk[curr_topic] -= 1
        self.m -= 1

        # Ensure count of tables per topic is some non-negative number
        assert self.mk[curr_topic] >= 0

        # Get rid of topic if dish isn't being served anywhere
        if self.mk[curr_topic] == 0:
            self.used_topics.remove(curr_topic)

    def get_wt_distribution(self, word):
        """
        Each word is assumed to be drawn from a distribution F(theta_ji). Given the word,
        get the distribution from word-topic counts.

        :param word: Current Word
        :return: Distribution for given word across all existing topics.
        """
        return [wt[word] for wt in self.wt] / self.nk

    def table_posterior(self, j, w_dist):
        """
        Calculate table posterior (to assign current word to new table).

        :param j: Current Document ID.
        :param w_dist: Distribution for current word over topics.
        :return: Normalized table posterior.
        """
        current_tables = self.used_tables[j]
        table_posterior = self.dt[j][current_tables] * w_dist[self.dk[j][current_tables]]
        new_table_prob = np.inner(self.mk, w_dist) + self.gamma / self.V
        table_posterior[0] = new_table_prob * self.alpha / (self.gamma + self.m)

        # Return normalized posterior
        return table_posterior / table_posterior.sum()

    def table_dish_posterior(self, j, t):
        """
        Calculate dish (topic) posterior distribution for the current table.

        :param j: Current Document Index
        :param t: Current Table
        :return: Dish (topic) posterior distribution
        """
        old_dish = self.dk[j][t]  # Table dish assignment --> 0 implies a removed dish (draw new)

        v_beta = self.V * self.beta
        gammaln_beta = gammaln(self.beta)
        nk = self.nk.copy()
        t_size = self.dt[j][t]  # Number seated at current table

        # Decrement table size from current topic counts
        nk[old_dish] -= t_size

        # Retrieve counts for each topic (after decrement)
        nk = nk[self.used_topics]

        # Calculate posterior
        log_dish_posterior = np.log(self.mk[self.used_topics]) + gammaln(nk) - gammaln(nk + t_size)
        log_dish_posterior_new = np.log(self.gamma) + gammaln(v_beta) - gammaln(v_beta - t_size)

        for word, t_count in self.dtw[j][t].iteritems():
            assert t_count >= 0
            if t_count == 0: continue
            wt_count = np.array([n.get(word, self.beta) for n in self.wt])
            wt_count[old_dish] -= t_count
            wt_count = wt_count[self.used_topics]
            wt_count[0] = 1  # Dummy value for log

            # Update dish posterior
            log_dish_posterior += gammaln(wt_count + t_count) - gammaln(wt_count)
            log_dish_posterior_new += gammaln(self.beta + t_count) - gammaln_beta

        # Exponentiate and normalize posterior
        log_dish_posterior[0] = log_dish_posterior_new

        dish_posterior = np.exp(log_dish_posterior - log_dish_posterior.max())
        return dish_posterior / dish_posterior.sum()

    def word_dish_posterior(self, w_dist):
        """
        Calculate dish (topic) posterior distribution for the current word.

        :param w_dist: Distribution for current word over topics.
        :return: Dish (topic) posterior multinomial distribution.
        """
        dish_posterior = (self.mk * w_dist)[self.used_topics]
        dish_posterior[0] = self.gamma / self.V  # Probability of drawing new dish (new topic)

        # Return normalized posterior
        return dish_posterior / dish_posterior.sum()

    def new_table(self, j, k):
        """
        Helper function to facilitate the addition of a new table for the current document.

        :param j: Current Document Index
        :param k: Sampled dish (topic) for new table
        :return: New table index
        """
        assert k in self.used_topics  # Check that topic has been properly initialized
        for new_t, t in enumerate(self.used_tables):
            if new_t != t:
                break
        else:  # Enters this only if loop exhausted -> not if break statement activated
            new_t = len(self.used_tables[j])
            self.dt[j].resize(new_t + 1)
            self.dk[j].resize(new_t + 1)
            self.dtw[j].append(None)

        # Create new table, set counts, etc.
        self.used_tables[j].insert(new_t, new_t)
        self.dt[j][new_t] = 0  # Unnecessary, but invariant
        self.dtw[j][new_t] = defaultdict(0)

        # Set new table dish to k
        self.dk[j][new_t] = k
        self.mk[k] += 1
        self.m += 1

        return new_t

    def new_dish(self):
        """
        Helper function to facilitate the creation of a new dish (topic).

        :return: New topic index
        """
        for new_k, k in enumerate(self.used_topics):
            if new_k != k:
                break
        else:  # Enters this only if loop exhausted -> not if break statement activated
            new_k = len(self.used_topics)
            if new_k >= len(self.wt):  # Resizing conditional
                self.nk = np.resize(self.nk, new_k + 1)
                self.mk = np.resize(self.mk, new_k + 1)
                self.wt.append(None)
            # Sanity checks
            assert new_k == self.used_topics[-1] + 1
            assert new_k < len(self.wt)

        # Insert new topic into list of used topics
        self.used_topics.insert(new_k, new_k)
        self.nk[new_k] = self.beta * self.V
        self.mk[new_k] = 0
        self.wt[new_k] = defaultdict(self.beta)

        return new_k

    def get_theta(self):
        """
        After conducting sampling, estimate theta, the probability of topic given
        document for each document in the corpus.

        :return: List of lists containing document - topic distribution.
        """
        td_dist = np.array(self.mk, dtype=float)
        td_dist[0] = self.gamma
        td_dist *= self.alpha / td_dist[self.used_topics].sum()

        theta = []
        for j, dt in enumerate(self.dt):
            topic_doc_dist = td_dist.copy()
            for t in self.used_tables[j]:
                if t == 0:
                    continue
                k = self.dk[j][t]
                topic_doc_dist[k] += dt[t]
            topic_doc_dist = topic_doc_dist[self.used_topics]
            theta.append(topic_doc_dist / topic_doc_dist.sum())

        return np.array(theta)

    def get_phi(self):
        """
        After conducting sampling, estimate phi, the probability of word given
        topic for each word in the vocabulary.

        :return: List of dictionaries mapping topic - word to probability.
        """
        return [defaultdict(self.beta / self.nk[k]).update(
            (word, word_count / self.nk[k]) for word, word_count in self.wt[k].iteritems())
                for k in self.used_topics if k != 0]

    def summary(self, fp=sys.stdout):
        """
        Summarize the topic model output, by printing the optimal topic number (K), as well
        as the distributions for word given topic (phi), and the distributions for topic given
        document (phi)

        :param fp: Location to write output
        """
        num_topics = len(self.used_topics) - 1
        k_map = dict((k, i - 1) for i, k in enumerate(self.used_topics))

        dish_count = np.zeros(num_topics, dtype=int)
        word_count = [defaultdict(0) for _ in xrange(num_topics)]

        for j, doc in enumerate(self.docs):
            for word, table in zip(doc, self.tables[j]):
                k = k_map[self.dk[j][table]]
                dish_count[k] += 1
                word_count[k][word] += 1

        phi = self.get_phi()
        for k, phi_k in enumerate(phi):
            fp.write("\n-- Topic: %d (%d Words)\n" % (self.used_topics[k + 1], dish_count[k]))
            for w in sorted(phi_k, key=lambda x: -phi_k[x])[:20]:
                fp.write("%s: %f (%d)\n" % (self.corpus.vocab_map[w], phi_k[w], word_count[k][w]))

        fp.write("--- Document-Topic Distribution\n")
        theta = self.get_theta()
        for j, theta_j in enumerate(theta):
            fp.write("%d\t%s\n" % (j, "\t".join("%.3f" % p for p in theta_j[1:])))



if __name__ == "__main__":
    character_corpus = Corpus(test_dict)
    hdp = HDP(character_corpus)
    hdp.summary()
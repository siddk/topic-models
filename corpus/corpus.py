"""
corpus.py

Core structure file defining a corpus class -> a series of documents consisting of
several words. Has all the necessary attributes for the suite of topic models.
"""

test_dict = {"Katniss": ['run', 'shoot', 'rescue', 'save'],
             "Batman": ['save', 'fight', 'rescue', 'stop'],
             "Harry": ['fly', 'fight', 'rescue', 'save'],
             "Voldemort": ['torture', 'scare', 'kill', 'capture'],
             "Joker": ['torture', 'kill', 'capture', 'kidnap'],
             "Snow": ['kill', 'torture', 'capture', 'laugh'],
             "Alfred": ['advise', 'support', 'console', 'teach'],
             "Haymitch": ['support', 'console', 'teach', 'protect'],
             "Dumbledore": ['rescue', 'protect', 'advise', 'support', 'teach']}


class Corpus:
    def __init__(self, docs):
        """
        Given a dictionary mapping document IDs to a list of words, build a Corpus.

        :param docs: Dictionary mapping IDs to List of Words.
        """
        # Dictionary Mapping Document IDs to Document Names
        self.doc_ids = {doc_id: i for i, doc_id in enumerate(docs.keys())}
        self.id_map = {i: doc_id for doc_id, i in self.doc_ids.items()}

        # Build Vocabulary
        self.vocab = {w: i for i, w in enumerate(set(reduce(lambda x, y: x + y, docs.values())))}
        self.vocab_map = {i: w for w, i in self.vocab.items()}

        # Build Final Representation using List of Lists
        self.docs = self.build_representation(docs)

    def build_representation(self, docs):
        """
        Build representation swapping document name for document id, word for vocab id.

        :return: List of lists representing entire corpus
        """
        internal_rep = {}
        for d in docs:
            internal_rep[self.doc_ids[d]] = map(lambda x: self.vocab[x], docs[d])
        return internal_rep


if __name__ == "__main__":
    test_corpus = Corpus(test_dict)
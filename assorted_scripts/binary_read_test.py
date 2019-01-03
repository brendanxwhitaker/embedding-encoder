from gensim.models.keyedvectors import KeyedVectors

import pyemblib
import gensim
import os

path = os.path.abspath("../../embeddings/GoogleNews-vectors-negative300.bin")

gensim_embeddings = KeyedVectors.load_word2vec_format('path/to/GoogleNews-vectors-negative300.bin', binary=True)
embedding = pyemblib.read(path, format='Word2Vec', mode=pyemblib.Mode.Binary)

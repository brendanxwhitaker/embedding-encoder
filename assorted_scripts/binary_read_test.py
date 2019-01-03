from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

import pyemblib
import gensim
import os

parent = os.path.abspath("../../embeddings/") 
path = os.path.abspath("../../embeddings/GoogleNews-vectors-negative300.bin")
glove = os.path.abspath("../../embeddings/glove.840B.300d.txt") 

# gensim working. 
# google_news = KeyedVectors.load_word2vec_format(path, binary=True)
glove2word2vec(glove_input_file=glove, word2vec_output_file=os.path.join(parent, "glove.840B.300d_Word2Vec_format.txt"))


# pyemblib not working. 
# embedding = pyemblib.read(path, format='Word2Vec', mode=pyemblib.Mode.Binary)

import pyemblib
import os

path = os.path.abspath("~/NER/embeddings/GoogleNews-vectors-negative300.bin")
embedding = pyemblib.read(path, format='Word2Vec', mode=pyemblib.Mode.Binary)

# embedding-encoder
Autoencoder for word2vec word embedding files.

The ideas is that we pick one (ora  few, this is "batch\_size"), and compute the distance from this embedding to all others, and train on this at each step. 


A placeholder is a stand-in for our dataset. We'll assign data to it at a later date. Data is "fed" into the persistent TensorFlow network graph through these placeholders. 

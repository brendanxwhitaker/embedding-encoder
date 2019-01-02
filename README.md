# embedding-encoder
Autoencoder for word2vec word embedding files.

### ae.py

Main script. Calls preprocessing.py and next\_batch.py. 
Given an embedding in .txt or .bin format, preproceses, and generates
in batches distance vectors. Uses single-hidden layer autoencoder to
compress distance vectors into shape of source embedding file. 
Saves the model and saves embedding vectors as text file. If the script
is run with a model name which already exists, it saves the embedding 
vectors instead of retraining.  


The ideas is that we pick one (ora  few, this is "batch\_size"), and compute the distance from this embedding to all others, and train on this at each step. 
A placeholder is a stand-in for our dataset. We'll assign data to it at a later date. Data is "fed" into the persistent TensorFlow network graph through these placeholders. 

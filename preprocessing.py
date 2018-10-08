import multiprocessing as mp
import pandas as pd
import numpy as np

from progressbar import progressbar
from tqdm import tqdm

import pyemblib
import scipy
import queue
import time
import sys
import os 

#========1=========2=========3=========4=========5=========6=========7==

def check_valid_dir(some_dir):
    if not os.path.isdir(some_dir):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")
        print("DIES IST EIN UNGÜLTIGES VERZEICHNIS!!!!")
        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()

#========1=========2=========3=========4=========5=========6=========7==

def check_valid_file(some_file):
    if not os.path.isfile(some_file):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")
        print("DIES IST KEIN GÜLTIGER SPEICHERORT FÜR DATEIEN!!!!")
        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: [numpy matrix of word vectors, df of the labels]
def process_embedding(emb_path, first_n):

    print("Preprocessing. ")
    file_name_length = len(emb_path)
    last_char = emb_path[file_name_length - 1]

    # Decide if it's a binary or text embedding file, and read in
    # the embedding as a dict object, where the keys are the tokens
    # (strings), and the values are the components of the corresponding 
    # vectors (floats).
    embedding = {}
    if (first_n != 0):
        if (last_char == 'n'):
            embedding = pyemblib.read(emb_path, 
                                      mode=pyemblib.Mode.Binary,
                                      first_n=first_n) 
        elif (last_char == 't'):
            embedding = pyemblib.read(emb_path, 
                                      mode=pyemblib.Mode.Text, 
                                      first_n=first_n)
        else:
            print("Unsupported embedding format. ")
            exit()
    else:
        if (last_char == 'n'):
            embedding = pyemblib.read(emb_path, 
                                      mode=pyemblib.Mode.Binary)
        elif (last_char == 't'):
            embedding = pyemblib.read(emb_path, 
                                      mode=pyemblib.Mode.Text) 
        else:
            print("Unsupported embedding format. ")
            exit()

    # convert embedding to pandas dataframe
    # "words_with_friends" is the column label for the vectors
    # this df has shape [num_inputs,2] since the vectors are all in 1
    # column as length d lists 
    emb_df = pd.Series(embedding, name="words_with_friends")
    # print(emb_df.head(10))

    # reset the index of the dataframe
    emb_df = emb_df.reset_index()
    # print(emb_df.head(10))

    # matrix of just the vectors
    emb_matrix = emb_df.words_with_friends.values.tolist()
    # print(emb_matrix[0:10])

    # dataframe of just the vectors
    vectors_df = pd.DataFrame(emb_matrix,index=emb_df.index)
    # print(vectors_df.head(10))

    # numpy matrix of just the vectors
    vectors_matrix = vectors_df.as_matrix()
    # print(vectors_matrix[0:10])

    return vectors_matrix, emb_df.loc[:,"index"]

#========1=========2=========3=========4=========5=========6=========7== 

from preprocessing  import process_embedding
from preprocessing  import check_valid_file
from preprocessing  import check_valid_dir
from next_batch     import next_batch

import tensorflow.contrib.layers    as lays
import multiprocessing              as mp
import tensorflow                   as tf 
import pandas                       as pd
import numpy                        as np

from progressbar    import progressbar
from tqdm           import tqdm

import datetime
import pyemblib
import scipy
import queue
import time
import sys
import os

'''
rand_vecs.py

Script to generate an embedding with random, normalized vectors for each
token from a source vocab file. 
'''

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: a tuple of the script arguments
def parse_args():

    emb_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        first_n = sys.argv[2]
    else:
        first_n = 0

    args = [emb_path,first_n]
    return args

#========1=========2=========3=========4=========5=========6=========7==

# VECTOR GENERATION FUNCTION
def epoch(  embedding_tensor,
            label_df,
            new_emb_path):
 
    name = mp.current_process().name
    print(name, 'Starting')
    sys.stdout.flush()

    # shape [<num_inputs>,<dimensions>]
    rand_emb_array = []

    for i in range(len(embedding_tensor)):
        vec = np.random.rand(len(embedding_tensor[0]))
        vec = vec / np.linalg.norm(vec)
        rand_emb_array.append(vec)

    print("labels shape: ", labels.shape)
    print("rand_emb_array shape: ", rand_emb_array.shape)
    
    # creates the emb dict
    dist_emb_dict = {}
    for i in tqdm(range(len(labels))):
        emb_array_row = rand_emb_array[i]
        dist_emb_dict.update({labels[i]:emb_array_row})

    # saves the embedding
    pyemblib.write(dist_emb_dict, 
                   new_emb_path, 
                   mode=pyemblib.Mode.Text)
 
    print(name, 'Exiting')
    return

#=========1=========2=========3=========4=========5=========6=========7=

def mkproc(func, arguments):
    p = mp.Process(target=func, args=arguments)
    p.start()
    return p

#========1=========2=========3=========4=========5=========6=========7==

def genflow(emb_path,first_n):

    print_sleep_interval = 1 
    check_valid_file(emb_path)

    source_name = os.path.splitext(os.path.basename(emb_path))[0]
    print("Source name:", source_name)

    # take the first n most frequent word vectors for a subset
    # set to 0 to take entire embedding
    first_n = 0

    # Preprocess. 
    vectors_matrix,label_df = process_embedding(emb_path, 
                                                first_n,
                                                None)

    # We get the dimensions of the input dataset. 
    shape = vectors_matrix.shape
    print("Shape of embedding matrix: ", shape)
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()

    # number of rows in the embedding 
    num_inputs = shape[0]
    num_outputs = num_inputs 

    # dimensionality of the embedding file
    num_hidden = shape[1]

    # clears the default graph stack
    tf.reset_default_graph()

    #===================================================================

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Reset dimensions for vocab subset
    shape = vectors_matrix.shape
    print("Shape of embedding matrix: ", shape)
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()

    # Reset
    num_inputs = shape[0]
     
    # we read the numpy array "vectors_matrix" into tf as a Tensor
    embedding_tensor = tf.constant(vectors_matrix)
    print("shape of emb_tens is: ", 
          embedding_tensor.get_shape().as_list())
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
     
    embedding_unshuffled = embedding_tensor
    emb_transpose_unshuf = tf.transpose(embedding_unshuffled)
    emb_transpose_unshuf = tf.cast(emb_transpose_unshuf, tf.float32)

    #===================================================================

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H:%M")
    
    # the name of the embedding to save
    # something like "~/<path>/steve.txt"
    check_valid_dir("../embeddings/")
    new_emb_path =  str("../embeddings/random__source--" + source_name 
                    + "__" + timestamp + ".txt")

    # RUN THE TRAINING PROCESS
    eval_process = mp.Process(name="eval",
                               target=epoch,
                               args=(embedding_unshuffled,
                                     label_df,
                                     new_emb_path))

    eval_process.start()    
    eval_process.join()

    return

#========1=========2=========3=========4=========5=========6=========7==

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here 
    
    args = parse_args()

    emb_path = args[0]
    first_n = args[1]   
 
    genflow(emb_path,first_n) 



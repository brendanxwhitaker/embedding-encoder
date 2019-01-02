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

import pyemblib
import scipy
import queue
import time
import sys
import os

'''
get_dist_vecs.py

Script to compute distance vectors for a set of vocab words from 
a pretrained embedding. 
'''

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: a tuple of the script arguments
def parse_args():

    emb_path = sys.argv[1]
    vocab_path = sys.argv[2]

    args = [emb_path,
            "/homes/3/whitaker.213/similarity_test/wordsim_vocab.txt"]

    return args

#========1=========2=========3=========4=========5=========6=========7==

# TRAINING FUNCTION
def epoch(  embedding_tensor,
            label_df,
            init,
            saver,
            model_path,
            new_emb_path,
            retrain,
            num_processes):
 
    name = mp.current_process().name
    print(name, 'Starting')
    sys.stdout.flush()
    with tf.Session() as sess:
         
        # initializes all the variables that have been created
        sess.run(init)

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

    while not batch_queue.empty():
        try:
            batch_queue.get(timeout=0.001)
        except:
            pass
 
    print(name, 'Exiting')
    return

#=========1=========2=========3=========4=========5=========6=========7=

def mkproc(func, arguments):
    p = mp.Process(target=func, args=arguments)
    p.start()
    return p

#========1=========2=========3=========4=========5=========6=========7==

def genflow(emb_path,vocab_path):

    print_sleep_interval = 1

    retrain = True
    
    check_valid_file(emb_path)

    with open(vocab_path, "r") as source:
        vocab = source.read().split('\n')

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

    # change vectors matrix to just the vocab
    vectors_matrix,label_df = process_embedding(emb_path, 
                                                first_n,
                                                None)

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
    
    # program hangs when I try to run from saved model    
    ''' 
    # Later, launch the model, use the saver to restore variables from 
    # disk, and do some work with the model.
    with tf.Session() as sess:
      
        # Restore variables from disk.
        saver.restore(sess, model_path)
        print("Model restored.")
        
    # Check the values of the variables
    print(embedding_tensor.shape)

    # hidden_out = hidden_layer.eval(feed_dict={X: })
    # for row in hidden_out:
        # print(row) 
    '''

    # the name of the embedding to save
    # something like "~/<path>/steve.txt"
    new_emb_path = "../embeddings/random_source-" + source_name + ".txt"

    # Saving embedding vectors file. 
    retrain = False

    # RUN THE TRAINING PROCESS
    eval_process = mp.Process(name="eval",
                               target=epoch,
                               args=(embedding_unshuffled,
                                     label_df,
                                     init,
                                     saver,
                                     new_emb_path,
                                     retrain))
    eval_process.start()    
    eval_process.join()

    return

#========1=========2=========3=========4=========5=========6=========7==

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here 
    
    args = parse_args()

    emb_path = args[0]
    vocab_path = args[1]
    
    genflow(emb_path,vocab_path) 



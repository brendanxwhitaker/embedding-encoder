import tensorflow.contrib.layers as lays
import multiprocessing as mp
import tensorflow as tf 
import pandas as pd
import numpy as np

from cuto import process_embedding
from cuto import check_valid_file
from cuto import check_valid_dir
from cuto import next_batch
from tqdm import tqdm

import pyemblib
import scipy
import time
import sys
import os 

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: a tuple of the script arguments
def parse_args():

    emb_path = sys.argv[1]
    nn_path = sys.argv[2]
    
    args = [emb_path,
            nn_path,
           ]
 
    return args

#========1=========2=========3=========4=========5=========6=========7==

# SAVE example
one_line_dict = exDict = {1:1, 2:2, 3:3}
df = pd.DataFrame.from_dict([one_line_dict])
df.to_csv('file.txt', header=False, index=True, mode='a')

#========1=========2=========3=========4=========5=========6=========7==

def runflow(emb_path,nn_path):

    check_valid_file(emb_path)
    if os.path.isfile(model_index_path):
        print("There is already a model saved with this name. ") 
        retrain = False
    
    vectors_matrix,label_df = process_embedding(emb_path)

    # We get the dimensions of the input dataset. 
    shape = vectors_matrix.shape
    print("Shape of embedding matrix: ", shape)

    # number of rows in the embedding 
    num_inputs = shape[0]
    num_outputs = num_inputs 

    # dimensionality of the embedding file
    num_hidden = shape[1]

    # clears the default graph stack
    tf.reset_default_graph()











    eval_batch_size = 100

    # HYPERPARAMETERS
    eval_num_batches = num_inputs // eval_batch_size # floor division
    print("Defining hyperparameters: ")
    print("Eval batch size: ", eval_batch_size)
    print("Number of batches: ", eval_num_batches)
    

    # we instantiate the queue
    seed2_queue = mp.Queue()  
    batch2_queue = mp.Queue()
 
    # So we need each Process to take from an input queue, and 
    # to output to an output queue. All 3 batch generation 
    # prcoesses will read from the same input queue, and what 
    # they will be reading is just an integer which corresponds 
    # to an iteration 
    for iteration in tqdm(range(eval_num_batches)):  
        seed2_queue.put(iteration)

    print("seed queue size: ", seed2_queue.qsize())


    # CREATE MATRIXMULT PROCESSES
    batch_d = mp.Process(name="batch_d",
                         target=next_batch,
                         args=(embedding_unshuffled,
                               emb_transpose_unshuf,
                               label_df,
                               eval_batch_size,
                               seed2_queue,
                               batch2_queue))
    
    batch_e = mp.Process(name="batch_e",
                         target=next_batch,
                         args=(embedding_unshuffled,
                               emb_transpose_unshuf,
                               label_df,
                               eval_batch_size,
                               seed2_queue,
                               batch2_queue))
    
    batch_f = mp.Process(name="batch_f",
                         target=next_batch,
                         args=(embedding_unshuffled,
                               emb_transpose_unshuf,
                               label_df,
                               eval_batch_size,
                               seed2_queue,
                               batch2_queue))

    print("About to start the batch processes. ")
    batch_d.start()
    batch_e.start()
    batch_f.start()

    # the name of the embedding to save
    # something like "~/<path>/steve.tt"
    save_path = "/homes/3/whitaker.213/eleven_embedding.txt"

    # RUN THE TRAINING PROCESS
    eval_process = mp.Process(name="eval",
                              target=create_emb,
                              args=(embedding_unshuffled,
                                    eval_num_batches,
                                    batch2_queue,
                                    hidden_layer,
                                    X,
                                    init,
                                    save_path))
    eval_process.start()    

    print("queue is full. ")
        
    batch_d.join()
    batch_e.join()
    batch_f.join()

    eval_process.join()

#========1=========2=========3=========4=========5=========6=========7==

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here 
    
    args = parse_args()

    emb_path = args[0]
    model_path = args[1]
    batch_size = args[2]
    epochs = args[3]
    learning_rate = args[4]
    keep_prob = args[5]
    num_processes = args[6]
    
    trainflow(emb_path,model_path,batch_size,epochs,
              learning_rate,keep_prob,num_processes) 

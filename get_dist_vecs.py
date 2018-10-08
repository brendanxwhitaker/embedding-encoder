import tensorflow.contrib.layers as lays
import multiprocessing as mp
import tensorflow as tf 
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

# RETURNS: a tuple of the script arguments
def parse_args():

    emb_path = sys.argv[1]
    model_path = sys.argv[2]
    # batch_size = int(sys.argv[3])
    # epochs = int(sys.argv[4])
    # learning_rate = float(sys.argv[5])
    # keep_prob = float(sys.argv[6])
    # num_processes = int(sys.argv[7])

    args = [emb_path,
            model_path,
            10, 
            50,
            0.001,
            0.5,
            3]

    return args

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

# NEXTBATCH FUNCTION
# Function which creates a new batch of size batch_size, randomly chosen
# from our dataset. For batch_size = 1, we are just taking one 100-dimen
# -sional vector and computing its distance from every other vector in 
# the dataset and then we have a num_inputs-dimensional vector which rep
# -resents the distance of every vector from our "batch" vector. If we 
# choose batch_size = k, then we would have k num_inputs-dimensional ve-
# ctors. 
def next_batch(entire_embedding,emb_transpose,label_df,
               batch_size,seed_queue,batch_queue):

    num_dimensions = int(entire_embedding.shape[1])
    name = mp.current_process().name
    print(name, 'Starting')
    sys.stdout.flush()
    with tf.Session() as sess: 
        
        slice_shape = [batch_size, num_dimensions]
 
        # Note slice_begin looks like "[row_loc, column_loc]", it is 
        # simply the coordinates of where we start our slice, so we 
        # set its placeholder to have shape(1,2)
        SLICE_BEGIN = tf.placeholder(tf.int32, shape=(2))
        slice_embedding = tf.slice(entire_embedding, 
                                   SLICE_BEGIN, slice_shape)
       
        # This is a placeholder for the output of the "slice_embedding"
        # operation. It outputs a slice of the embedding, with 
        # shape "slice_shape". 
        SLICE_OUTPUT = tf.placeholder(tf.float32,shape=slice_shape)
        mult = tf.matmul(SLICE_OUTPUT,emb_transpose)

        # just need a value for "iteration" that is not -1 to satisfy
        # while condition on first loop
        iteration = 0 
        
        while True:
            while batch_queue.qsize() > 10:
                time.sleep(1)                  
            
            iteration = seed_queue.get()
            print("Iteration: ", iteration) 
            
            if iteration == -1:
                batch_queue.put([-1,-1])
                break

            current_index = iteration * batch_size 
            dist_row_list = []
    
            # get the corresponding slice of the labels as df
            slice_df = label_df.iloc[current_index:
                                     current_index + batch_size]
            # slice_df = pd.DataFrame([0,0])
            # begin the slice at the "current_index"-th row in
            # the first column
            slice_begin = [current_index, 0]
        
            # slice the embedding from "slice_begin" with shape
            # "slice_shape"
            slice_output = sess.run(slice_embedding, 
                                    feed_dict={
                                     SLICE_BEGIN:slice_begin
                                    }
                                   )
          
            # take dot product of slice with embedding
            dist_matrix = sess.run(mult, 
                                   feed_dict={
                                    SLICE_OUTPUT:slice_output
                                   }
                                  ) 
            sys.stdout.flush()
            
            # dist_matrix has shape 
            batch_queue.put([dist_matrix,slice_df])
            if iteration > 997:
                print("pushed batch")
        
    print(name, 'Exiting')
    sys.stdout.flush()
    return

#========1=========2=========3=========4=========5=========6=========7==

# TRAINING FUNCTION
def epoch(embedding_tensor,num_batches,step,batch_queue,train,
          loss,loss_vectors,hidden_layer,X,init,saver,model_path,
          new_emb_path,retrain):
 
    name = mp.current_process().name
    print(name, 'Starting')
    sys.stdout.flush()
    with tf.Session() as sess:
         
        # initializes all the variables that have been created
        sess.run(init)
        
        # list of slices which compose the new embedding
        embedding_slices = []
        label_slices = []

        # just can't be -1
        batch = np.zeros((5,5))
        total_error = 0
        batches_completed = 0
        print("number of batches: ", num_batches)
        
        while True:

            batch_loss = 0
            batch,slice_df = batch_queue.get()
            
            # break for halt batch
            # be careful not to check for np.array but for np.ndarray!
            if not isinstance(batch, np.ndarray):
                print("Found the halt batch. ") 
                batch,slice_df = batch_queue.get()
                batch,slice_df = batch_queue.get()
                break 
            print("Batches completed: ", batches_completed) 
            batches_completed = batches_completed + 1
            sys.stdout.flush()

            if retrain:
                sess.run(train,feed_dict={X: batch})
                err_vectors = loss_vectors.eval(feed_dict={X:batch})
            for j in range(len(err_vectors)):
                # get the loss value for the jth distance vector
                # in the batch
                err_vector = err_vectors[j] 
                # print("errvector shape,",err_vector.shape)
                
                # convert shape from (n,1) to (1,n)
                err_vector = np.asarray([err_vector])
                
                # get the sum of the loss over that distance vector
                loss_val = np.sum(err_vector)
                
                # add to total loss for entire vocab
                total_error += loss_val
                batch_loss += loss_val
            
            # when we put "batch" in the feed dict, it uses it 
            # wherever there is an "X" in the definition of "loss" OR
            # in the definition of any tf function that "loss" calls.  
            # err = loss.eval(feed_dict={X: batch})
            # print("\tLoss:", err)
        
            with open("loss_log_20K.txt","a") as f:
                f.write(str(batch_loss) + "\n")
            else: 
                # slice of the output from the hidden layer
                hidden_out_slice = hidden_layer.eval(feed_dict={X: batch})
                embedding_slices.append(hidden_out_slice)

                # add the slice of labels that corresponds to the batch
                label_slices.append(slice_df)

        if retrain:
                    
            ''' 
            print("Printing total loss. ")
            with open("loss_log_20K.txt","a") as f:
                f.write("Total Loss for epoch " 
                        + str(step) + ": " + str(total_error) + "\n")
            '''

            # save_path = saver.save(sess,"../model_small.ckpt")
            save_path = saver.save(sess,model_path)
            print("Model saved in path: %s" % save_path)
        else:
 
            # makes dist_emb_array a 3-dimensional array 
            dist_emb_array = np.stack(embedding_slices)
            
            # concatenates the first dimension, so dist_emb_array has 
            # shape [<num_inputs>,<dimensions>]
            dist_emb_array = np.concatenate(dist_emb_array)

            # concatenates the list of pands Series containing the words
            # that correspond to the new vectors in "dist_emb_array"
            labels = pd.concat(label_slices)
            print("labels shape: ", labels.shape)
            print("dist_emb_array shape: ", dist_emb_array.shape)
            
            # creates the emb dict
            dist_emb_dict = {}
            for i in tqdm(range(len(labels))):
                emb_array_row = dist_emb_array[i]
                dist_emb_dict.update({labels[i]:emb_array_row})

            # saves the embedding
            pyemblib.write(dist_emb_dict, 
                           save_path, 
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

def trainflow(emb_path,model_path,batch_size,epochs,
              learning_rate,keep_prob,num_processes):

    print_sleep_interval = 1

    model_index_path = model_path + ".index"

    retrain = True

    check_valid_file(emb_path)
    if os.path.isfile(model_index_path):

        print("There is already a model saved with this name. ") 
        time.sleep(print_sleep_interval)           
        sys.stdout.flush()
        retrain = False

    # take the first $n$ most frequent word vectors for a subset
    # set to 0 to take entire embedding
    first_n = 0
   
    vectors_matrix,label_df = process_embedding(emb_path, first_n)

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

    print("Learning rate is: ",learning_rate)
    time.sleep(print_sleep_interval)               
    sys.stdout.flush()
    
    # probability of outputting nonzero value in dropout layer. So the 
    # input to the dropout layer goes to zero 1 - keep_prob of the time 
    print("Dropout layer keep_prob is: ", keep_prob)
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()

    # HYPERPARAMETERS
    num_batches = num_inputs // batch_size # floor division
    print("Defining hyperparameters: ")
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
    print("Epochs: ", epochs)
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
    print("Batch size: ", batch_size)
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
    print("Number of batches: ", num_batches)                  
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()

    # clears the default graph stack
    tf.reset_default_graph()

    # PLACEHOLDER
    # "tf.float32" just means the data type is an integer. The shape is 
    # in the form [<columns>,<rows>], and "None" means it can be any 
    # value. So this placeholder can have any number of rows, and must 
    # have "num_inputs" columns. 
    print("Initializing placeholder. ")
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
    X = tf.placeholder(tf.float32, shape=[None, num_inputs])

    # WEIGHTS
    print("Initializing weights. ")
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
    # we use a variance scaling initializer so that it is capable of 
    # adapting its scale to the shape of the weight tensors. 
    initializer = tf.variance_scaling_initializer()
    input_weights = tf.Variable(initializer([num_inputs, num_hidden]), 
                                dtype=tf.float32)
    output_weights = tf.Variable(initializer([num_hidden, num_outputs]), 
                                 dtype=tf.float32)

    # BIAS
    input_bias = tf.Variable(tf.zeros(num_hidden))
    output_bias = tf.Variable(tf.zeros(num_outputs))

    # ACTIVATION
    act_func = tf.nn.relu

    print("Initializing layers and defining loss function. ")
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()

    #===================================================================

    # LAYERS
    # the argument of act_func is a Tensor, and the variable 
    # "hidden_layer" itself is also a Tensor. This hidden layer is just 
    # going to compute the element-wise relu 

    hidden_layer = act_func(tf.matmul(X, input_weights) + input_bias)

    # With probability keep_prob, outputs the input element scaled up 
    # by 1 / keep_prob, otherwise outputs 0. The scaling is so that the 
    # expected sum is unchanged.
    dropout_layer = tf.nn.dropout(hidden_layer,keep_prob=keep_prob)
    output_layer = tf.matmul(dropout_layer,output_weights)+output_bias 

    # We define our loss function, minimize MSE
    loss_vectors = tf.abs(output_layer - X)
    reduce_mean = tf.reduce_mean(X)
    loss = tf.reduce_mean(tf.abs(output_layer - X))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # UNIT NORM THE EMBEDDING
    print("Unit norming the embedding. ")
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
    norms_matrix = np.linalg.norm(vectors_matrix, axis=1)
    norms_matrix[norms_matrix==0] = 1
    vectors_matrix = vectors_matrix / np.expand_dims(norms_matrix, -1)
    print(vectors_matrix.shape)

    # we read the numpy array "vectors_matrix" into tf as a Tensor
    embedding_tensor = tf.constant(vectors_matrix)
    print("shape of emb_tens is: ", 
          embedding_tensor.get_shape().as_list())
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
     
    embedding_unshuffled = embedding_tensor
    emb_transpose_unshuf = tf.transpose(embedding_unshuffled)
    emb_transpose_unshuf = tf.cast(emb_transpose_unshuf, tf.float32)
    emb_transpose = tf.transpose(embedding_tensor)
    emb_transpose = tf.cast(emb_transpose, tf.float32)

    #===================================================================

    with open("loss_log_20K.txt","a") as f:
        f.write("\n")
        f.write("=====================================================")
        f.write("\n")

    # this is where we'll add the dataset shuffler
    tf.random_shuffle(embedding_tensor)                    
   
    if retrain:
 
        for step in tqdm(range(epochs)):
            print("this is the ", step, "th epoch.")


            # we instantiate the queue
            seed_queue = mp.Queue()  
            
            mananger = mp.Manager()
            batch_queue = mananger.Queue()
         
            # So we need each Process to take from an input queue, and 
            # to output to an output queue. All 3 batch generation 
            # prcoesses will read from the same input queue, and what 
            # they will be reading is just an integer which corresponds 
            # to an iteration 
            for iteration in tqdm(range(num_batches)):  
                seed_queue.put(iteration)

            # put in "p" halt seeds to tell the processes when to end
            for i in range(3):
                seed_queue.put(-1)

            new_emb_path = ""

            # CREATE MATRIXMULT PROCESSES
            batch_args = (embedding_tensor,
                          emb_transpose,
                          label_df,
                          batch_size,
                          seed_queue,
                          batch_queue) 
            print("About to start the batch processes. ")
            allprocs = [mkproc(next_batch, batch_args) 
                        for x in range(num_processes)]

            # RUN THE TRAINING PROCESS
            train_process = mp.Process(name="train",
                                       target=epoch,
                                       args=(embedding_tensor,
                                             num_batches,
                                             step,
                                             batch_queue,
                                             train,
                                             loss,
                                             loss_vectors,
                                             hidden_layer,
                                             X,
                                             init,
                                             saver,
                                             model_path,
                                             new_emb_path,
                                             retrain))
            train_process.start() 

            print("queue is full. ")

            # join the processes, i.e. end them
            for process in allprocs:
                process.terminate()

            # join the processes, i.e. end them
            for process in allprocs:
                process.join()
                
            print("batch generation functions joined. ")

            train_process.join()

            print("train joined. ")

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
    batch_args = (embedding_unshuffled,
                  emb_transpose_unshuf,
                  label_df,
                  eval_batch_size,
                  seed2_queue,
                  batch2_queue))
    print("About to start the batch processes. ")
    allprocs = [mkproc(next_batch, batch_args) 
                for x in range(num_processes)]

    # the name of the embedding to save
    # something like "~/<path>/steve.txt"
    new_emb_path = "/homes/3/whitaker.213/eleven_embedding.txt"

    retrain = False

    # RUN THE TRAINING PROCESS
    eval_process = mp.Process(name="eval",
                               target=epoch,
                               args=(embedding_unshuffled,
                                     eval_num_batches,
                                     step,
                                     batch2_queue,
                                     train,
                                     loss,
                                     loss_vectors,
                                     hidden_layer,
                                     X,
                                     init,
                                     saver,
                                     model_path,
                                     new_emb_path,
                                     retrain))
    eval_process.start()    

    print("queue is full. ")
        
    # join the processes, i.e. end them
    for process in allprocs:
        process.terminate()

    # join the processes, i.e. end them
    for process in allprocs:
        process.join()

    eval_process.join()

    return

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



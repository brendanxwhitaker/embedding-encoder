import tensorflow.contrib.layers as lays
import multiprocessing as mp
import tensorflow as tf 
import pandas as pd
import numpy as np

from tqdm import tqdm

import pyemblib
import scipy
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
            1,
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

def process_embedding(emb_path):

    print("Preprocessing. ")
    file_name_length = len(emb_path)
    last_char = emb_path[file_name_length - 1]

    # Decide if it's a binary or text embedding file, and read in
    # the embedding as a dict object, where the keys are the tokens
    # (strings), and the values are the components of the corresponding 
    # vectors (floats).
    embedding = {}
    if (last_char == 'n'):
        embedding = pyemblib.read(emb_path, mode=pyemblib.Mode.Binary)
    elif (last_char == 't'):
        embedding = pyemblib.read(emb_path, mode=pyemblib.Mode.Text)
    else:
        print("Unsupported embedding format. ")
        exit()

    # convert embedding to pandas dataframe 
    emb_df = pd.Series(embedding, name="words_with_friends")

    # reset the index of the dataframe
    emb_df = emb_df.reset_index()

    # matrix of just the vectors
    emb_matrix = emb_df.words_with_friends.values.tolist()

    # dataframe of just the vectors
    vectors_df = pd.DataFrame(emb_matrix,index=emb_df.index)

    # numpy matrix of just the vectors
    vectors_matrix = vectors_df.as_matrix()

    return vectors_matrix

#========1=========2=========3=========4=========5=========6=========7== 

# NEXTBATCH FUNCTION
# Function which creates a new batch of size batch_size, randomly chosen
# from our dataset. For batch_size = 1, we are just taking one 100-dimen
# -sional vector and computing its distance from every other vector in 
# the dataset and then we have a num_inputs-dimensional vector which rep
# -resents the distance of every vector from our "batch" vector. If we 
# choose batch_size = k, then we would have k num_inputs-dimensional ve-
# ctors. 
def next_batch(entire_embedding,emb_transpose,
               batch_size,input_queue,output_queue):

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
        # "slice_shape" rows and the same number of columns as 
        # "entire_embedding".  
        SLICE_OUTPUT = tf.placeholder(tf.float32, 
                                      shape=slice_shape)
        mult = tf.matmul(SLICE_OUTPUT,emb_transpose)

        while not input_queue.empty():     
            iteration = input_queue.get()
            print("Iteration: ", iteration) 
            current_index = iteration * batch_size 
            dist_row_list = []
            for i in tqdm(range(batch_size)):

                # begin the slice at the "current_index"-th row in
                # the first column
                slice_begin = [current_index, 0]
            
                # we sum the products of each element in the row axis 
                # of both matrices.

                # slice the embedding from "slice_begin" with shape
                # "slice_shape"
                slice_output = sess.run(slice_embedding, 
                                        feed_dict={
                                         SLICE_BEGIN:slice_begin
                                        }
                                       )
    
                # take dot product of slice with embedding
                dist_row = sess.run(mult, 
                                    feed_dict={
                                     SLICE_OUTPUT:slice_output
                                    }
                                   )
                 
                sys.stdout.flush()
                dist_row_list.append(dist_row[0])
                current_index = current_index + 1
           
            # used to be doing this with tf.stack(), changing to numpy 
            # beacuse fuck that. 
            dist_matrix = np.stack(dist_row_list)
            sys.stdout.flush()
            output_queue.put(dist_matrix)
        
    print(name, 'Exiting')
    sys.stdout.flush()
    return

#========1=========2=========3=========4=========5=========6=========7==

# TRAINING FUNCTION
def epoch(embedding_tensor,num_batches,step,output_queue,
          train,loss,hidden_layer,X,init,saver,model_path):
    
    name = mp.current_process().name
    print(name, 'Starting')
    sys.stdout.flush()
    with tf.Session() as sess:
        
        # initializes all the variables that have been created
        sess.run(init)

        batches_completed = 0
        while(batches_completed < num_batches):
            batch = output_queue.get()
            sess.run(train,feed_dict={X: batch})
            batches_completed = batches_completed + 1                    

        if step % 1 == 0:
            err = loss.eval(feed_dict={X: batch})
            print(step, "\tLoss:", err)
            
            # changing what is being fed to the dict from 
            # vectors_matrix to embedding_tensor
            output2d = hidden_layer.eval(feed_dict={X: batch})

        # this line still must be modified
        # output2dTest = 
        # hidden_layer.eval(feed_dict={X: batch})

        # save_path = saver.save(sess,"../model_small.ckpt")
        save_path = saver.save(sess,model_path)
        print("Model saved in path: %s" % save_path)
    print(name, 'Exiting')
    return

#========1=========2=========3=========4=========5=========6=========7==

# EMBEDDING GENERATION FUNCTION
def create_emb(embedding_tensor,num_batches,
               output_queue,hidden_layer,X,init,save_path):
    
    name = mp.current_process().name
    print(name, 'Starting')
    sys.stdout.flush()
    with tf.Session() as sess:
        
        # initializes all the variables that have been created
        # sess.run(init)

        batches_completed = 0
        while(batches_completed < num_batches):
            batch = output_queue.get()
            hidden_layer

            hidden_out_slice = hidden_layer.eval(feed_dict={X: batch})
            batches_completed = batches_completed + 1                    
            print("shape of hidden out: ", hidden_out_slice.shape)
    

    print(name, 'Exiting')
    return

#========1=========2=========3=========4=========5=========6=========7==

def trainflow(emb_path,model_path,batch_size,epochs,
              learning_rate,keep_prob,num_processes):

    model_index_path = model_path + ".index"

    retrain = True

    check_valid_file(emb_path)
    if os.path.isfile(model_index_path):
        print("There is already a model saved with this name. ") 
        retrain = False
    
    vectors_matrix = process_embedding(emb_path)

    # We get the dimensions of the input dataset. 
    shape = vectors_matrix.shape
    print("Shape of embedding matrix: ", shape)

    # number of rows in the embedding 
    num_inputs = shape[0]
    num_outputs = num_inputs 

    # dimensionality of the embedding file
    num_hidden = shape[1]

    print("Learning rate is: ",learning_rate)
    
    # probability of outputting nonzero value in dropout layer. So the 
    # input to the dropout layer goes to zero 1 - keep_prob of the time 
    print("Dropout layer keep_prob is: ", keep_prob)

    # HYPERPARAMETERS
    num_batches = num_inputs // batch_size # floor division
    print("Defining hyperparameters: ")
    print("Epochs: ", epochs)
    print("Batch size: ", batch_size)
    print("Number of batches: ", num_batches)

    # clears the default graph stack
    tf.reset_default_graph()

    # PLACEHOLDER
    # "tf.float32" just means the data type is an integer. The shape is 
    # in the form [<columns>,<rows>], and "None" means it can be any 
    # value. So this placeholder can have any number of rows, and must 
    # have "num_inputs" columns. 
    print("Initializing placeholder. ")
    X = tf.placeholder(tf.float32, shape=[None, num_inputs])

    # WEIGHTS
    print("Initializing weights. ")
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

    #===================================================================

    # LAYERS
    # the argument of act_func is a Tensor, and the variable 
    # "hidden_layer" itself is also a Tensor. This hidden layer is just 
    # going to compute the element-wise relu 

    # should the input to the hidden layer not be X, should it instead
    # have shape [None,num_dimensions]?
    hidden_layer = act_func(tf.matmul(X, input_weights) + input_bias)

    # With probability keep_prob, outputs the input element scaled up 
    # by 1 / keep_prob, otherwise outputs 0. The scaling is so that the 
    # expected sum is unchanged.
    dropout_layer = tf.nn.dropout(hidden_layer,keep_prob=keep_prob)
    output_layer = tf.matmul(dropout_layer,output_weights)+output_bias 

    # We define our loss function, minimize MSE
    loss = tf.reduce_mean(tf.abs(output_layer - X))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # UNIT NORM THE EMBEDDING
    print("Unit norming the embedding. ")
    norms_matrix = np.linalg.norm(vectors_matrix, axis=1)
    norms_matrix[norms_matrix==0] = 1
    vectors_matrix = vectors_matrix / np.expand_dims(norms_matrix, -1)
    print(vectors_matrix.shape)

    # we read the numpy array "vectors_matrix" into tf as a Tensor
    embedding_tensor = tf.constant(vectors_matrix)
    print("shape of emb_tens is: ", 
          embedding_tensor.get_shape().as_list())

    emb_transpose = tf.transpose(embedding_tensor)
    emb_transpose = tf.cast(emb_transpose, tf.float32)

    #===================================================================
   
    if retrain:
 
        for step in range(epochs):
            print("this is the ", step, "th epoch.")

            # this is where we'll add the dataset shuffler
            tf.random_shuffle(embedding_tensor)

            # we instantiate the queue
            input_queue = mp.Queue()  
            output_queue = mp.Queue()
         
            # So we need each Process to take from an input queue, and 
            # to output to an output queue. All 3 batch generation 
            # prcoesses will read from the same input queue, and what 
            # they will be reading is just an integer which corresponds 
            # to an iteration 
            for iteration in tqdm(range(num_batches)):  
                input_queue.put(iteration)
         
            # CREATE MATRIXMULT PROCESSES
            batch_a = mp.Process(name="batch_a",
                                 target=next_batch,
                                 args=(embedding_tensor,
                                       emb_transpose,
                                       batch_size,
                                       input_queue,
                                       output_queue))
            
            batch_b = mp.Process(name="batch_b",
                                 target=next_batch,
                                 args=(embedding_tensor,
                                       emb_transpose,
                                       batch_size,
                                       input_queue,
                                       output_queue))
            
            batch_c = mp.Process(name="batch_c",
                                 target=next_batch,
                                 args=(embedding_tensor,
                                       emb_transpose,
                                       batch_size,
                                       input_queue,
                                       output_queue))

            print("About to start the batch processes. ")
            batch_a.start()
            batch_b.start()
            batch_c.start()

            # RUN THE TRAINING PROCESS
            train_process = mp.Process(name="train",
                                       target=epoch,
                                       args=(embedding_tensor,
                                             num_batches,
                                             step,
                                             output_queue,
                                             train,
                                             loss,
                                             hidden_layer,
                                             X,
                                             init,
                                             saver,
                                             model_path))
            train_process.start()    

            print("queue is full. ")
                
            batch_a.join()
            batch_b.join()
            batch_c.join()

    #===================================================================
     
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

    eval_batch_size = 200

    # HYPERPARAMETERS
    eval_num_batches = num_inputs // eval_batch_size # floor division
    print("Defining hyperparameters: ")
    print("Eval batch size: ", eval_batch_size)
    print("Number of batches: ", eval_num_batches)

    # we instantiate the queue
    input_queue = mp.Queue()  
    output_queue = mp.Queue()
 
    # So we need each Process to take from an input queue, and 
    # to output to an output queue. All 3 batch generation 
    # prcoesses will read from the same input queue, and what 
    # they will be reading is just an integer which corresponds 
    # to an iteration 
    for iteration in tqdm(range(num_batches)):  
        input_queue.put(iteration)
 
    # CREATE MATRIXMULT PROCESSES
    batch_a = mp.Process(name="batch_a",
                         target=next_batch,
                         args=(embedding_tensor,
                               emb_transpose,
                               eval_batch_size,
                               input_queue,
                               output_queue))
    
    batch_b = mp.Process(name="batch_b",
                         target=next_batch,
                         args=(embedding_tensor,
                               emb_transpose,
                               eval_batch_size,
                               input_queue,
                               output_queue))
    
    batch_c = mp.Process(name="batch_c",
                         target=next_batch,
                         args=(embedding_tensor,
                               emb_transpose,
                               eval_batch_size,
                               input_queue,
                               output_queue))

    print("About to start the batch processes. ")
    batch_a.start()
    batch_b.start()
    batch_c.start()

    # RUN THE TRAINING PROCESS
    eval_process = mp.Process(name="eval",
                              target=evaluate,
                              args=(embedding_tensor,
                                    num_batches,
                                    output_queue,
                                    hidden_layer,
                                    X,
                                    init,
                                    save_path))
    train_process.start()    

    print("queue is full. ")
        
    batch_a.join()
    batch_b.join()
    batch_c.join()





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



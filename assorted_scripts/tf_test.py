import tensorflow.contrib.layers as lays
import multiprocessing as mp
import tensorflow as tf 
import pandas as pd
import numpy as np
import progressbar
import pyemblib
import scipy
import sys 

print("Done with imports. ")

#=========1=========2=========3=========4=========5=========6=========7=

num_inputs = 1000


print("Defining hyperparameters:")
# MORE HYPERPARAMETERS
epochs = 1  
batch_size = 10
num_batches = num_inputs // batch_size #floor division
batches_at_a_time = 3

print("Epochs: ", epochs)
print("Batch size: ", batch_size)
print("Number of batches: ", num_batches)







# we instantiate the queue
input_queue = mp.Queue()  
output_queue = mp.Queue()

# So we need each Process to take from an input queue, and to 
# output to an output queue. All 3 batch generation prcoesses
# will read from the same input queue, and what they will be 
# reading is just an integer which corresponds to an iteration 
for iteration in progressbar.progressbar(
range(num_batches)):  
    input_queue.put(iteration)     



entire_embedding = np.zeros((1000,100))


# we read the numpy array "embedding_matrix" into tf as a Tensor
embedding_tensor = tf.constant(entire_embedding)
print(
"shape of emb_tens is: ",embedding_tensor.get_shape().as_list())

matrix_queue = mp.Queue() 
batch_size = 10
iteration = 0
emb_transpose = tf.transpose(embedding_tensor)

print("emb_transpose shape: ",emb_transpose.shape)
emb_transpose = tf.cast(emb_transpose, tf.float32) 


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

    name = mp.current_process().name
    print(name, 'Starting')
    sys.stdout.flush()
    with tf.Session() as sess:
    
#=========1=========2=========3=========4=========5=========6=========7= 
       

        # slice_size is a constant, should have 
        # "entire_embedding.shape[1] = 100"
        slice_size = [1,100]

        print("The shape of slice_size is: ", np.array(slice_size).shape)
 
        # Note slice_begin is an array with 1 row and 2 columns below,
        # so we set its placeholder to have shape(1,2)
        SLICE_BEGIN = tf.placeholder(tf.int32, shape=(2))
        slice_embedding = tf.slice(entire_embedding, SLICE_BEGIN, slice_size)
       
        # This is a placeholder for the output of the "slice_embedding"
        # operation. It outputs a slice of the embedding, with 
        # "slice_size" rows and the same number of columns as 
        # "entire_embedding". So we get that number by taking
        # "entire_embedding.shape[1]". 

        print(entire_embedding.shape[1])


        SLICE_OUTPUT = tf.placeholder(tf.float32, shape=(slice_size[0],int(entire_embedding.shape[1])))
        mult = tf.matmul(SLICE_OUTPUT,emb_transpose)

        # in case I want to change it back to a tf.stack() operation
        #DIST_ROW_LIST = tf.placeholder(tf.float32, shape=(batch_size, 
        #stack = tf.stack(

        while not input_queue.empty():
            
            iteration = input_queue.get()
            print("Iteration: ", iteration) 
            current_index = iteration * batch_size 
            dist_row_list = []
            for i in progressbar.progressbar(range(batch_size)):

                slice_begin = [current_index,0]
                # we sum the products of each element in the row axis of 
                # both matrices.
                
                # the commented out line below should work, but I'm going to try and split it into two. 
                #dist_row = sess.run(mult, feed_dict={SLICE_OUTPUT:sess.run(slice_embedding, feed_dict={SLICE_BEGIN:slice_begin})}) 
                slice_output = sess.run(slice_embedding, feed_dict={SLICE_BEGIN:slice_begin})
                dist_row = sess.run(mult, feed_dict={SLICE_OUTPUT:slice_output})
                 
                # Above line is just a dot product
                #print("dist_row shape is: ",dist_row.shape)
                sys.stdout.flush()
                dist_row_list.append(dist_row[0])
                current_index = current_index + 1
           
            # print("dist_row_list shape is: ",dist_row_list.shape)
            
            # used to be doing this with tf.stack(), changing to numpy beacuse fuck that. 
            dist_matrix = np.stack(dist_row_list)
            #print("dist_matrix shape is: ",dist_matrix.shape)
            sys.stdout.flush()
            output_queue.put(dist_matrix)
        
    print(name, 'Exiting')
    sys.stdout.flush()
    return






next_batch(entire_embedding,emb_transpose,batch_size,input_queue,output_queue)

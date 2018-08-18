import tensorflow.contrib.layers as lays
import multiprocessing as mp
import tensorflow as tf 
import pandas as pd
import numpy as np

import progressbar
import pyemblib
import scipy
import sys 

#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: the embedding as a dict object, where the keys are the tokens
#          (strings), and the values are the components of the
#          corresponding vectors (floats).

def read_embedding():

    in_file = sys.argv[1]
    file_name_length = len(in_file)
    last_char = in_file[file_name_length - 1]

    # Decide if it's a binary or text embedding file, and read in
    embedding = {}
    if (last_char == 'n'):
        embedding = pyemblib.read(in_file, mode=pyemblib.Mode.Binary)
    elif (last_char == 't'):
        embedding = pyemblib.read(in_file, mode=pyemblib.Mode.Text)
    else:
        print("Unsupported embedding format. ")
        exit()

    return embedding

#=========1=========2=========3=========4=========5=========6=========7=

def process_embedding(embedding):

    print("Preprocessing. ")

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

#=========1=========2=========3=========4=========5=========6=========7=


print("Setting parameters.")

# PARAMETERS   
num_outputs = num_inputs 
learning_rate = 0.001
print("Learning rate is: ",learning_rate)
# probability of outputting nonzero value in dropout layer. So the input
# to the dropout layer goes to zero 1 - keep_prob of the time.  
keep_prob = 0.5
print("Dropout layer keep_prob is: ", keep_prob)
# Clears the default graph stack and resets the global default graph.
# (graph as in the network graph)
tf.reset_default_graph()

# PLACEHOLDER
# a placeholder is a stand-in for our dataset. We'll assign data to it
# at a later date. Data is "fed" into the network graph through these
# placeholders. "tf.float32" just means the data type is an integer. 
# the shape is in the form [<columns>,<rows>], and "None" means it can
# be any value. So this placeholder can have any number of rows, and 
# must have num_inputs columns. 
print("Initializing placeholder. ")
X = tf.placeholder(tf.float32, shape=[None, num_inputs])

#=========1=========2=========3=========4=========5=========6=========7=

# WEIGHTS
print("Initializing weights. ")
# we use a variance scaling initializer so that it is capable of adapti-
# ng its scale to the shape of the weight tensors. 
initializer = tf.variance_scaling_initializer()
input_weights = tf.Variable(
initializer([num_inputs, num_hidden]), dtype=tf.float32)
output_weights = tf.Variable(
initializer([num_hidden, num_outputs]), dtype=tf.float32)

# BIAS
input_bias = tf.Variable(tf.zeros(num_hidden))
output_bias = tf.Variable(tf.zeros(num_outputs))

# ACTIVATION
act_func = tf.nn.relu

#=========1=========2=========3=========4=========5=========6=========7=

print("Initializing layers and defining loss function. ")

# LAYERS
# the argument of act_func is a Tensor, and the variable "hidden_layer"
# itself is also a Tensor. This hidden layer is just going to compute
# the element-wise relu of "tf.matmul(X, input_weights) + input_bias)". 

# Note matmul is just the matrix multiplication of X and input_weights,
# then we're adding input_bias, the bias variable, which is initialized
# to zeroes. 
hidden_layer = act_func(tf.matmul(X, input_weights) + input_bias)

# With probability keep_prob, outputs the input element scaled up by 
# 1 / keep_prob, otherwise outputs 0. The scaling is so that the expect-
# ed sum is unchanged.
dropout_layer = tf.nn.dropout(hidden_layer,keep_prob=keep_prob)
output_layer = tf.matmul(dropout_layer, output_weights) + output_bias 

# We define our loss function, minimize MSE
# right now we are using abs instead of square, does this matter?
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

#=========1=========2=========3=========4=========5=========6=========7=

# we read the numpy array "vectors_matrix" into tf as a Tensor
embedding_tensor = tf.constant(vectors_matrix)
print(
"shape of emb_tens is: ",embedding_tensor.get_shape().as_list())

batch_size = 10
iteration = 0
emb_transpose = tf.transpose(embedding_tensor)
emb_transpose = tf.cast(emb_transpose, tf.float32)

#=========1=========2=========3=========4=========5=========6=========7=

# MORE HYPERPARAMETERS
print("Defining hyperparameters:")
epochs = 1  
batch_size = 10
num_batches = num_inputs // batch_size #floor division
batches_at_a_time = 3

print("Epochs: ", epochs)
print("Batch size: ", batch_size)
print("Number of batches: ", num_batches)

#=========1=========2=========3=========4=========5=========6=========7= 

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
       
        # slice_size is a constant, should have 
        # "entire_embedding.shape[1] = 100"
        slice_size = [1,100]
 
        # Note slice_begin is an array with 1 row and 2 columns below,
        # so we set its placeholder to have shape(1,2)
        SLICE_BEGIN = tf.placeholder(tf.int32, shape=(2))
        slice_embedding = tf.slice(entire_embedding, 
                                   SLICE_BEGIN, slice_size)
       
        # This is a placeholder for the output of the "slice_embedding"
        # operation. It outputs a slice of the embedding, with 
        # "slice_size" rows and the same number of columns as 
        # "entire_embedding". So we get that number by taking
        # "entire_embedding.shape[1]". 
        SLICE_OUTPUT = tf.placeholder(tf.float32, 
                                      shape=(slice_size[0], 
                                      entire_embedding.shape[1]))
        mult = tf.matmul(SLICE_OUTPUT,emb_transpose)

        while not input_queue.empty():     
            iteration = input_queue.get()
            print("Iteration: ", iteration) 
            current_index = iteration * batch_size 
            dist_row_list = []
            for i in progressbar.progressbar(range(batch_size)):

                slice_begin = [current_index,0]
            
                # we sum the products of each element in the row axis 
                # of both matrices.
                
                # the commented out stuff below should work, but I'm 
                # going to try and split it into two. 
                #dist_row = sess.run(
                #                    mult, 
                #                    feed_dict={
                #                     SLICE_OUTPUT:sess.run(
                #                      slice_embedding, 
                #                      feed_dict={
                #                       SLICE_BEGIN:slice_begin
                #                      }
                #                     )
                #                    }
                #                   ) 
                slice_output = sess.run(slice_embedding, 
                                        feed_dict={
                                         SLICE_BEGIN:slice_begin
                                        }
                                       )
                dist_row = sess.run(mult, 
                                    feed_dict={
                                     SLICE_OUTPUT:slice_output
                                    }
                                   )
                 
                # Above line is just a dot product
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

#=========1=========2=========3=========4=========5=========6=========7=

# TRAINING FUNCTION
def train_epoch(embedding_tensor,num_batches,batch_size,
output_queue,train,hidden_layer):
    
    name = mp.current_process().name
    print(name, 'Starting')
    sys.stdout.flush()
    with tf.Session() as sess:
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

        save_path = saver.save(sess,"../model.ckpt")
        print("Model saved in path: %s" % save_path)
    print(name, 'Exiting')
    return

#=========1=========2=========3=========4=========5=========6=========7=

def train():

    for step in range(epochs):
        print("this is the ", step, "th epoch.")

        # this is where we'll add the dataset shuffler
        tf.random_shuffle(embedding_tensor)

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
     
        # this used to be num_batches // batches_at_a_time, wrong?
        # CREATE MATRIXMULT PROCESSES
        batch_a = mp.Process(name="batch_a",target=next_batch,
        args=(embedding_tensor,emb_transpose,batch_size,
        input_queue,output_queue))
        
        batch_b = mp.Process(name="batch_b",target=next_batch,
        args=(embedding_tensor,emb_transpose,batch_size,
        input_queue,output_queue))

        batch_c = mp.Process(name="batch_c",target=next_batch,
        args=(embedding_tensor,emb_transpose,batch_size,
        input_queue,output_queue))

        print("About to start the batch processes. ")
        batch_a.start()
        batch_b.start()
        batch_c.start()

        # RUN THE TRAINING PROCESS
        train_epoch = mp.Process(name="train",target=train_epoch,args=
        (embedding_tensor,num_batches,
        batch_size,output_queue,train,hidden_layer))
        train_epoch.start()   

        print("queue is full. ")
            
        batch_a.join()
        batch_b.join()
        batch_c.join()

#=========1=========2=========3=========4=========5=========6=========7=

def main():
    
    embedding = read_embedding()

    vectors_matrix = process_embedding(embedding)

    # We get the dimensions of the input dataset. 
    shape = vectors_matrix.shape
    print("Shape of embedding matrix: ", shape)

    # number of rows in the embedding 
    num_inputs = shape[0]

    # dimensionality of the embedding file
    num_hidden = shape[1]



if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()



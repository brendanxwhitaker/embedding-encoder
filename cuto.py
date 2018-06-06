import tensorflow.contrib.layers as lays
import tensorflow as tf 
import pandas as pd
import numpy as np
import progressbar
import pyemblib
import scipy
import sys 

#=========1=========2=========3=========4=========5=========6=========7=
# The idea is that we pick one (or a few, this is your batch_size), and
# compute the distance from this embedding to all others, and train on
# this at each step. it's getting angry that there is a column containi-
# ng text (the words) so we need to create a copy of the numpy array wh-
# ich has this column cut off and use only the real-valued columns as
# inputs. UPDATE: this has been taken care of. 

# We allow the user to specify the dataset. It is the first and only
# argument of this python script.

# READIN 
in_file = sys.argv[1]
file_name_length = len(in_file)
last_char = in_file[file_name_length - 1]

# Decide if it's a binary or text embedding file, and read in
# accordingly. 
if (last_char == 'n'):
    embedding = pyemblib.read(in_file, mode=pyemblib.Mode.Binary)
elif (last_char == 't'):
    embedding = pyemblib.read(in_file, mode=pyemblib.Mode.Text)
else:
    print("Unsupported embedding format. ")
    exit() 

#=========1=========2=========3=========4=========5=========6=========7=

#PREPROCESSING
# We convert the embedding dictionary file into a pandas dataframe.
# The dict object embeddding is converted to a dataframe. The keys are
# strings and the values are 100 dimensional vectors as list objects. We
# add an index to the dataframe, and then convert the list column to 100
# real-valued columns. Then we create a copy with only the index and the
# vector columns. 
embedding_series = pd.Series(embedding, name="words_with_friends")
emb_with_index = embedding_series.reset_index()
int_list = list(range(1, 101))
emb_vect_matrix = emb_with_index.words_with_friends.values.tolist()
real_val_embs = pd.DataFrame(
emb_vect_matrix,index=emb_with_index.index)
embedding_matrix = real_val_embs.as_matrix()

# We get the dimensions of the input dataset. 
shape = embedding_matrix.shape
print(shape)
# this is the number of rows in the dataset, i.e. the number of unique
# words in the embedding. 
num_inputs = shape[0]
#dimensionality of the embedding file
num_hidden = shape[1]

#=========1=========2=========3=========4=========5=========6=========7=
# Everything above this line is pretty simple and probably doesn't have
# to be reviewed any time soon. 
#=========1=========2=========3=========4=========5=========6=========7=

# PARAMETERS   
num_outputs = num_inputs 
learning_rate = 0.001
# probability of outputting nonzero value in dropout layer. So the input
# to the dropout layer goes to zero 1 - keep_prob of the time.  
keep_prob = 0.5
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
X = tf.placeholder(tf.float32, shape=[None, num_inputs])

#=========1=========2=========3=========4=========5=========6=========7=

# WEIGHTS
# we use a variance scaling initializer so that it is capable of adapti-
# ng its scale to the shape of the weight tensors. 
initializer = tf.variance_scaling_initializer()
input_weights= tf.Variable(
initializer([num_inputs, num_hidden]), dtype=tf.float32)
output_weights = tf.Variable(
initializer = tf.variance_scaling_initializer()
input_weights= tf.Variable(
initializer([num_inputs, num_hidden]), dtype=tf.float32)
output_weights = tf.Variable(
initializer([num_hidden, num_outputs]), dtype=tf.float32)

# BIAS
input_bias = tf.Variable(tf.zeros(num_hidden))
output_bias = tf.Variable(tf.zeros(num_outputs))

# ACTIVATION
act_func = tf.nn.relu

#=========1=========2=========3=========4=========5=========6=========7=

# LAYERS
# the argument of act_func is a Tensor, and the variable "hidden_layer"
# itself is also a Tensor. This hidden layer is just going to compute
$ the element-wise relu of "tf.matmul(X, input_weights) + input_bias)". 

# Note matmul is just the matrix multiplication of X and input_weights,
# then we're adding input_bias, the bias variable, which is initialized
# to zeroes. 
hidden_layer = act_func(tf.matmul(X, input_weights) + input_bias)

# With probability keep_prob, outputs the input element scaled up by 
# 1 / keep_prob, otherwise outputs 0. The scaling is so that the expect-
# ed sum is unchanged.
dropout_layer= tf.nn.dropout(hidden_layer,keep_prob=keep_prob)
output_layer = tf.matmul(dropout_layer, output_weights) + output_bias 

# We define our loss function, minimize MSE
# right now we are using abs instead of square, does this matter?
loss = tf.reduce_mean(tf.abs(output_layer - X))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

# NEXTBATCH FUNCTION
# Function which creates a new batch of size batch_size, randomly chosen
# from our dataset. For batch_size = 1, we are just taking one 100-dimen
# -sional vector and computing its distance from every other vector in 
# the dataset and then we have a num_inputs-dimensional vector which rep
# -resents the distance of every vector from our "batch" vector. If we 
# choose batch_size = k, then we would have k num_inputs-dimensional ve-
# ctors. 
def next_batch(entire_embedding,batch_size,iteration):

#=========1=========2=========3=========4=========5=========6=========7=

    # recall that entire_embedding is the input to this function, which
    # will always be just the entire dataset. And
    # embedding_matrix.shape[0] should just return the number of columns
    # in the dataset, in this case ~600000. 

    current_index = iteration * batch_size 
    # slice_size is a constant
    slice_size = [1,100]
    dist_row_list = []
    for i in range(batch_size):

        slice_begin = [current_index,0]
        # we sum the products of each element in the row axis of both
        # matrices.
        dist_row = tf.tensordot(
        tf.slice(entire_embedding,slice_begin,slice_size),
        entire_embedding,[[1],[1]]) # dot product
        print("dist_row shape is: ",dist_row.shape)
        dist_row_list.append(dist_row[0])
        current_index = current_index + 1
   
    # print("dist_row_list shape is: ",dist_row_list.shape)
    dist_matrix = tf.stack(dist_row_list)
    print("dist_matrix shape is: ",dist_matrix.shape)    
    return dist_matrix

# UNIT NORM THE EMBEDDING
norms_matrix = np.linalg.norm(embedding_matrix, axis=1)
norms_matrix[norms_matrix==0] = 1
embedding_matrix = embedding_matrix / np.expand_dims(norms_matrix, -1)
#embedding_matrix = np.sum(np.abs(embedding_matrix)**2,axis=-1)**(1./2)
print(embedding_matrix.shape)

#=========1=========2=========3=========4=========5=========6=========7=

# we read the numpy array "embedding_matrix" into tf as a Tensor
embedding_tensor = tf.constant(embedding_matrix)
print(
"shape of embedding_tensor is: ",embedding_tensor.get_shape().as_list())

# TESTING OF NEXTBATCH FUNCTION  
batch_size = 2
iteration = 0
test_batch = next_batch(
embedding_tensor,batch_size,iteration,dist_matrix)
print(test_batch.shape)
num_batches = num_inputs // batch_size #floor division

#=========1=========2=========3=========4=========5=========6=========7=

# TRAINING
with tf.Session() as sess:
    sess.run(init)
    for step in range(epochs):
        print("this is the ", step, "th epoch.")

        # this is where we'll add the dataset shuffler
        tf.random_shuffle(embedding_tensor)

        #"iteration" measures how far through the epoch we are. 
        for iteration in progressbar.progressbar(range(num_batches)):  
            batch = next_batch(
            embedding_tensor,batch_size,iteration,dist_matrix)
            sess.run(train,feed_dict={X: batch.eval()})
        
        if step % 1 == 0:
            err = loss.eval(feed_dict={X: embedding_matrix})
            print(step, "\tLoss:", err)
            output2d = hidden_layer.eval(
            feed_dict={X: embedding_matrix})

    #this line still must be modified
    #output2dTest = hidden_layer.eval(feed_dict={X: scaled_test_data})

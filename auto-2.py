import sys
import tensorflow.contrib.layers as lays
import tensorflow as tf 
import pandas as pd
import numpy as np
import pyemblib
import progressbar
import scipy

# the idea is that we pick one (or a few, this is your batchSize), and compute
# the distance from this embedding to all others, and train on this at each step. 
# it's getting angry that there is a column containing text (the words) so we need
# to create a copy of the numpy array which has this column cut off and use only 
# the real-valued columns as inputs. UPDATE: this has been taken care of. 

# we allow the user to specify the dataset. It is the first and only
# argument of this python script. 
inFile = sys.argv[1]

filenameLen = len(inFile)

lastChar = inFile[filenameLen - 1]

# decide if it's a binary or text embedding file, and read in accordingly. 
if (lastChar == 'n'):
    embedding = pyemblib.read(inFile, mode=pyemblib.Mode.Binary)
elif (lastChar == 't'):
    embedding = pyemblib.read(inFile, mode=pyemblib.Mode.Text)
else:
    print("Unsupported embedding format. ")
    exit()  

# we convert the embedding dictionary file into a pandas dataframe. First the
# dict object embeddding is converted to a dataframe. The keys  are  strings
# and the values are 100 dimensional vectors as list objects. We add an index 
# to the dataframe, and then convert the list column to 100 real-valued 
# columns. Then we create a copy with only the index and the vector columns. 

embeddingDF = pd.Series(embedding, name="wordsWithFriends")
embWithIndex = embeddingDF.reset_index()
intList = list(range(1, 101))
embVectMatrix = embWithIndex.wordsWithFriends.values.tolist()
realValEmbs = pd.DataFrame(embVectMatrix, index=embWithIndex.index)
data = realValEmbs.as_matrix()

# We get the dimensions of the input dataset. 
shape = data.shape
print(shape)
# this is the number of rows in the dataset, i.e. the number of unique words
# in the embedding. 
numInputs = shape[0]
#dimensionality of the embedding file
numHidden = shape[1]






#==========================================================================================
# Everything above this line is pretty simple and probably doesn't have to be reviewed
# any time soon. 
#==========================================================================================





# PARAMETERS   
numOutputs = numInputs 
learningRate = 0.001
# probability of outputting nonzero value in dropout layer. So the input
# to the dropout layer goes to zero 1 - keepProb of the time.  
keepProb = 0.5
# Clears the default graph stack and resets the global default graph.
# (graph as in the network graph)
tf.reset_default_graph()


# PLACEHOLDER
# a placeholder is a stand-in for our dataset. We'll assign data to it
# at a later date. Data is "fed" into the network graph through these
# placeholders. "tf.float32" just means the data type is an integer. 
# the shape is in the form [<columns>,<rows>], and "None" means it can
# be any value. So this placeholder can have any number of rows, and 
# must have numInputs columns. 
X = tf.placeholder(tf.float32, shape=[None, numInputs])

# WEIGHTS
# we use a variance scaling initializer so that it is capable of 
# adapting its scale to the shape of the weight tensors. 
initializer = tf.variance_scaling_initializer()
w = tf.Variable(initializer([numInputs, numHidden]), dtype=tf.float32)
wOut = tf.Variable(initializer([numHidden, numOutputs]), dtype=tf.float32)

# BIAS
b = tf.Variable(tf.zeros(numHidden))
bOut = tf.Variable(tf.zeros(numOutputs))



'''



# ACTIVATION
actFunc = tf.nn.relu

# LAYERS
# the argument of actFunc is a Tensor, and the variable "hiddenLayer"
# itself is also a Tensor. 
# this hidden layer is just going to compute the element-wise relu
# of "tf.matmul(X, w) + b)". 
# Note matmul is just the matrix multiplication of X and w, then 
# we're adding b, the bias variable, which is initialized to zeroes. 
hiddenLayer = actFunc(tf.matmul(X, w) + b)
# With probability keep_prob, outputs the input element scaled up by 
# 1 / keep_prob, otherwise outputs 0. The scaling is so that the 
# expected sum is unchanged.
dropoutLayer= tf.nn.dropout(hiddenLayer,keep_prob=keepProb)
outputLayer = tf.matmul(dropoutLayer, wOut) + bOut 


# define our loss function, minimize MSE
# right now we are using abs instead of square, does this matter?
loss = tf.reduce_mean(tf.abs(outputLayer - X))
optimizer = tf.train.AdamOptimizer(learningRate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()


# function which creates a new batch of size batchSize, randomly chosen
# from our dataset. For batchSize = 1, we are just taking one 100-dimen
# -sional vector and computing its distance from every other vector in 
# the dataset and then we have a numInputs-dimensional vector which rep
# -resents the distance of every vector from our "batch" vector. If we 
# choose batchSize = k, then we would have k numInputs-dimensional vect
# -ors. 
def nextBatch(xData,batchSize,iteration,distMatrix):

    # recall that xData is the input to this function, which will 
    # always be just the entire dataset. And data.shape[0] should
    # just return the number of columns in the dataset, in this case
    # ~600000. 

    # OKAY so we want to NOT use a random index generator here, and 
    # instead, at the beginning, we'll randomly shuffle our rows and 
    # iterate over them, and when we've reached the end, we know we've
    # gone through a single training epoch. So we'll have to implement
    # that above. 

    currentIndex = iteration * batchSize 
    sliceBegin = [currentIndex,0]
    sliceSize = [batchSize,100]
    batchRows = tf.slice(xData,sliceBegin,sliceSize)
    for i in range(batchSize):
        # picks the next single row vector from our batchRows
        batchSliceBegin = [i,0]
        batchSliceSize = [1,100]
        batchVector = tf.slice(batchRows,batchSliceBegin,batchSliceSize)
        # we sum the products of each element in the row axis of both
        # matrices.
        # print("batchVector shape is: ",batchVector.shape)
        # print("xData shape is: ",xData.shape)
        distRow = tf.tensordot(batchVector,xData,[[1],[1]]) # dot product
        # print("distRow shape is: ",distRow.shape)
        # print("distMatrix shape is: ",distMatrix.shape)
        distMatrix = tf.concat([distMatrix, distRow], 0) 
        # print("shape of distMatrix",distMatrix.shape)
    
    distMatrix = tf.slice(distMatrix,[1,0],[batchSize,numInputs])    
    return distMatrix


# we unit norm each vector in the dataset.
normsMatrix = np.linalg.norm(data, axis=1)
normsMatrix[normsMatrix==0] = 1
data = data / np.expand_dims(normsMatrix, -1)
#data = np.sum(np.abs(data)**2,axis=-1)**(1./2)
print(data.shape)
# we read the numpy array "data" into tf as a Tensor
dataTensor = tf.constant(data)
print("shape of dataTensor is: ",dataTensor.get_shape().as_list())


dummyDistVect = np.zeros(numInputs)
dummyDistArray = np.array([dummyDistVect])
distMatrix = tf.constant(dummyDistArray)



# some test code to see if the nextBatch function is working.  
batchSize = 2
iteration = 0
XBatch = nextBatch(dataTensor,batchSize,iteration,distMatrix)

print(XBatch.shape)



# we train the encoder/decoder.
numSteps = 10    #pretty sure this is the same as number of epochs
batchSize = 1
numBatches = numInputs // batchSize #floor division


with tf.Session() as sess:
    sess.run(init)
    for step in range(numSteps):
        print("this is the ", step, "th epoch.")
        # this is where we'll add the dataset shuffler
        tf.random_shuffle(dataTensor)
        #"iteration" measures how far through the epoch we are. 
        for iteration in progressbar.progressbar(range(numBatches)):  
            XBatch = nextBatch(dataTensor,batchSize,iteration,distMatrix)
            sess.run(train,feed_dict={X: XBatch.eval()})
        
        if step % 1 == 0:
            err = loss.eval(feed_dict={X: data})
            print(step, "\tLoss:", err)
            output2d = hiddenLayer.eval(feed_dict={X: data})
    #this line still must be modified
    #output2dTest = hiddenLayer.eval(feed_dict={X: scaled_test_data})


'''







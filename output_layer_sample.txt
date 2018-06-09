initializer = tf.variance_scaling_initializer()
w = tf.Variable(initializer([numInputs, numHidden]), dtype=tf.float32)
wOut = tf.Variable(initializer([numHidden, numOutputs]), dtype=tf.float32)

# BIAS
b = tf.Variable(tf.zeros(numHidden))
bOut = tf.Variable(tf.zeros(numOutputs))

# ACTIVATION
actFunc = tf.nn.relu

hiddenLayer = actFunc(tf.matmul(X, w) + b)
dropoutLayer= tf.nn.dropout(hiddenLayer,keep_prob=keepProb)
outputLayer = tf.matmul(dropoutLayer, wOut) + bOut




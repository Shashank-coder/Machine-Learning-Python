# Digit Recognition with 3 hidden layers
import tensorflow as tf

'''
input > weights > hidden layer1 (activation function) > weights > hidden layer2 (activation function) > weights > hidden layer3 (activation function) > output layer

compare output to intended output > cost function (cross entropy)
optimization function (optimmizer) > minimize cost(AdamOptmizer, Adaboost...)

backpropagation

feed forward + backpropagation = epoch
'''
from tensorflow.examples.tutorials.mnist import input_data
sess = tf.Session()
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# specifying no. of nodes in each hiiden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# specifying no. of nodes in output layer 0-9
n_classes = 10
# batch size specifies no. of samples the program will handle at a time
batch_size = 100

# placeholders are variables which had to be initialised at runtime
# height X width
# 28x28 pixels photo = 784
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):


    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                        'biases':tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases

    l1 = tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases']
    l1 = tf.nn.relu(l1)

    l2 = tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases']
    l2 = tf.nn.relu(l2)

    l3 = tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases']
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # cost is our cost function and optimizer minimizes the cost function using AdamOptmizer or Gradient Descent

    # no. of iterations -> forward + backpropagation
    hm_epochs = 15

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(mnist.train.num_examples/batch_size):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        # correct counts the acurate predictions and accuracy calculates the fraction and gives the output in float but this is the graph
        # in print statement we initialize the variables as test samples and so we get the accuracy on test samples
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)

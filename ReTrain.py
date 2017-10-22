import vgg16
import Paths
import tensorflow as tf


class ReTrain:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        directory = Paths.this_directory()

        self.sess = tf.Session()
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])

        self.vgg = vgg16.vgg16(imgs, directory + '/pre_trained/vgg16_weights.npz', self.sess, skip_last_layer=True)

        # with self.sess.graph.as_default():
        with tf.name_scope('VGG16'):
            self.add_last_layer(8)

        self.vgg.probs = tf.nn.softmax(self.vgg.fc8l, name='final_tensor')
        tf.summary.histogram('activations', self.vgg.probs)

    def add_last_layer(self, num_categories):
        with tf.name_scope('fc8') as scope:
            fc8w = tf.Variable(tf.truncated_normal([4096, num_categories],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc8b = tf.Variable(tf.constant(1.0, shape=[num_categories], dtype=tf.float32),
                               trainable=True, name='biases')
            self.vgg.fc8l = tf.nn.bias_add(tf.matmul(self.vgg.fc7, fc8w), fc8b)
            self.vgg.parameters += [fc8w, fc8b]

    def add_final_training_ops(self, class_count, final_tensor_name, bottleneck_tensor,
                               bottleneck_tensor_size):
        """Adds a new softmax and fully-connected layer for training.

        We need to retrain the top layer to identify our new classes, so this function
        adds the right operations to the graph, along with some variables to hold the
        weights, and then sets up all the gradients for the backward pass.

        The set up for the softmax and fully-connected layers is based on:
        https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

        Args:
          class_count: Integer of how many categories of things we're trying to
          recognize.
          final_tensor_name: Name string for the new final node that produces results.
          bottleneck_tensor: The output of the main CNN graph.
          bottleneck_tensor_size: How many entries in the bottleneck vector.

        Returns:
          The tensors for the training and cross entropy results, and tensors for the
          bottleneck input and ground truth input.
        """
        with tf.name_scope('input'):
        #     bottleneck_input = tf.placeholder_with_default(
        #         bottleneck_tensor,
        #         shape=[None, bottleneck_tensor_size],
        #         name='BottleneckInputPlaceholder')
        #
            ground_truth_input = tf.placeholder(tf.float32,
                                                [None, class_count],
                                                name='GroundTruthInput')

        # Organizing the following ops as `final_training_ops` so they're easier
        # to see in TensorBoard
        layer_name = 'final_training_ops'
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                initial_value = tf.truncated_normal(
                    [bottleneck_tensor_size, class_count], stddev=0.001)

                layer_weights = tf.Variable(initial_value, name='final_weights')

                self.variable_summaries(layer_weights)
            with tf.name_scope('biases'):
                layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
                self.variable_summaries(layer_biases)
            # with tf.name_scope('Wx_plus_b'):
            #     logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            #     tf.summary.histogram('pre_activations', logits)

        # final_tensor = tf.nn.softmax(logits, name=final_tensor_name)


        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=ground_truth_input, logits=logits)
            with tf.name_scope('total'):
                cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            train_step = optimizer.minimize(cross_entropy_mean)

        return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
                final_tensor)

    def add_evaluation_step(self, result_tensor, ground_truth_tensor):
        """Inserts the operations we need to evaluate the accuracy of our results.

        Args:
          result_tensor: The new final node that produces results.
          ground_truth_tensor: The node we feed ground truth data
          into.

        Returns:
          Tuple of (evaluation step, prediction).
        """
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                prediction = tf.argmax(result_tensor, 1)
                correct_prediction = tf.equal(
                    prediction, tf.argmax(ground_truth_tensor, 1))
            with tf.name_scope('accuracy'):
                evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)
        return evaluation_step, prediction

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

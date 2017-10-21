import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize

from pre_trained.imagenet_classes import class_names
import Paths
import vgg16

from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants
from tensorflow.python.platform import gfile


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        tf.import_graph_def(graph_def, name="from_pb_file")
    return graph


def save_graph_to_file(sess, graph, graph_file_name):
    output_graph_def = convert_variables_to_constants(
        sess, graph.as_graph_def(), ['final_tensor'])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return


def save_vgg_to_pb_file():
    directory = Paths.this_directory()

    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg16.vgg16(imgs, directory + '/pre_trained/vgg16_weights.npz', sess)

    save_graph_to_file(sess, sess.graph, 'out/output_graph.pb')
    # This is for humans
    with gfile.FastGFile('out/output_labels.txt', 'w') as f:
        f.write('\n'.join(class_names) + '\n')


if __name__ == '__main__':

    save_vgg_to_pb_file()

    # load a .pb file and try an image
    directory = Paths.this_directory()
    graph = load_graph(directory + '/out/output_graph.pb')

    with tf.Session(graph=graph) as sess:

        writer = tf.summary.FileWriter("output", sess.graph)
        writer.close()

        img2 = imread('Tests/Images/jet.jpg', mode='RGB')
        img2 = imresize(img2, (224, 224))

        final_op = sess.graph.get_tensor_by_name('from_pb_file/final_tensor:0')
        feed = {sess.graph.get_tensor_by_name('from_pb_file/Placeholder:0'): [img2]}

        prob = sess.run(final_op, feed_dict=feed)[0]
        preds = (np.argsort(prob)[::-1])[0:1]

        print(class_names[preds[0]], prob[preds[0]])

        assert abs(prob[preds[0]] - 0.972493) < 0.000001

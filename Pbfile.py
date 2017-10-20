import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize

from pre_trained.imagenet_classes import class_names
import Paths


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        tf.import_graph_def(graph_def, name="from_pb_file")
    return graph


if __name__ == '__main__':
    # load a .pb file and try and image
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
        for p in preds:
            print(class_names[p], prob[p])

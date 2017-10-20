import tensorflow as tf

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


import argparse
import numpy as np

from pre_trained.imagenet_classes import class_names
from scipy.misc import imread, imresize

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="out/output_graph.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions


    with tf.Session(graph=graph) as sess:

        writer = tf.summary.FileWriter("output", sess.graph)
        writer.close()

        img2 = imread('Tests/Images/jet.jpg', mode='RGB')
        img2 = imresize(img2, (224, 224))

        final_op = sess.graph.get_tensor_by_name('prefix/final_tensor:0')
        feed = {sess.graph.get_tensor_by_name('prefix/Placeholder:0'): [img2]}
        prob = sess.run(final_op, feed_dict=feed)[0]
        preds = (np.argsort(prob)[::-1])[0:1]
        for p in preds:
            print(class_names[p], prob[p])

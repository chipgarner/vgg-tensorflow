import vgg16
import tensorflow as tf
from scipy.misc import imread, imresize
import pytest
import Paths


def test_finds_the_weasel():
    directory = Paths.this_directory()
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])

    vgg = vgg16.vgg16(imgs, directory + '/pre_trained/vgg16_weights.npz', sess)

    img1 = imread(directory + '/Tests/Images/laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]

    assert prob[356] == pytest.approx(0.693235, 0.000001)

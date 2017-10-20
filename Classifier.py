import vgg16
import tensorflow as tf
from pre_trained.imagenet_classes import class_names
from scipy.misc import imread, imresize
import numpy as np
import Paths
from PIL import Image


# Classify a list of input images of any size
class Classifier:
    def __init__(self):
        directory = Paths.this_directory()

        self.sess = tf.Session()
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.vgg = vgg16.vgg16(imgs, directory + '/pre_trained/vgg16_weights.npz', self.sess)

    def classifier(self, images_in):
        images = self.resize_images(images_in)

        self.show_images(images_in)
        self.show_images(images)

        self.classify_images(images)

        writer = tf.summary.FileWriter("output", self.sess.graph)
        writer.close()

    def resize_images(self, images_in):
        images = []
        for image in images_in:
            images.append(imresize(image, (224, 224)))
        return images

    def classify_images(self, images):
        for image in images:
            probs, preds = self.classify(image)
            for p in preds:
                print(class_names[p], probs[p])

    def classify(self, image):
        return_top = 5
        probabilities = self.sess.run(self.vgg.probs, feed_dict={self.vgg.imgs: [image]})[0]
        predictions = np.argsort(probabilities)[::-1][0:return_top]
        return probabilities, predictions

    def show_images(self, images):
        for image in images:
            img = Image.fromarray(image, 'RGB')
            img.show()


if __name__ == '__main__':
    directory = Paths.this_directory()

    img1 = imread(directory + '/Tests/Images/laska.png', mode='RGB')
    img2 = imread(directory + '/Tests/Images/jet.jpg', mode='RGB')
    img3 = imread(directory + '/Tests/Images/Ship.jpg', mode='RGB')

    images = [img1, img2, img3]

    ifier = Classifier()
    ifier.classifier(images)

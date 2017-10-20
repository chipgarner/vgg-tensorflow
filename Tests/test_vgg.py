import vgg16


def test_init():
    imgs = 2
    network = vgg16.vgg16(imgs)

    assert network.conv1_2.get_shape() == (1, 1, 1, 64)
    assert network.probs.get_shape() == (1, 1000)

import vgg16
import Paths

def test_init():
    imgs = 2
    network = vgg16.vgg16(imgs)

    assert network.conv1_2.get_shape() == (1, 1, 1, 64)
    assert network.probs.get_shape() == (1, 1000)

def test_paths():
    dir = Paths.this_directory()
    print(dir)

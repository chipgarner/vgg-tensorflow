import ReTrain
import tensorflow as tf


def test_init_loads_vgg():
    re_train = ReTrain.ReTrain()

    final_tensor = re_train.sess.graph.get_tensor_by_name('final_tensor:0')
    assert final_tensor.shape[1] == 8

    tensor = re_train.sess.graph.get_tensor_by_name('VGG16/conv5_3/Conv2D:0')
    assert tensor.shape[1] == 14

#
# def test_add_bottleneck():
#     re_train = ReTrain.ReTrain()
#
#     re_train.add_bottleneck(re_train.graph)
#
# def test_training_operations_aded():
#     re_train = ReTrain.ReTrain()
#
#     class_count = 10
#     bottleneck_tensor_name = 'from_pb_file/VGG16/fc8/BiasAdd:0'
#     final_tensor_name = 'from_pb_file/final_tensor:0'
#     bottleneck_tensor = re_train.graph.get_tensor_by_name(bottleneck_tensor_name)
#     bottleneck_tensor_size = 1000

    # re_train.add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor,
    #                            bottleneck_tensor_size)

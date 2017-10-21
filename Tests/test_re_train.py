import ReTrain


def test_init():
    re_train = ReTrain.ReTrain()

    final_tensor = re_train.graph.get_tensor_by_name('from_pb_file/final_tensor:0')
    assert final_tensor.shape[1] == 1000

    tensor = re_train.graph.get_tensor_by_name('from_pb_file/VGG16/conv5_3/Conv2D:0')
    assert tensor.shape[1] == 14

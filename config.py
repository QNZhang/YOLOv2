class Config:

    ####### Dataset
    training_dir = "/home/mmv/Documents/3.datasets/VOCdevkit/"
    testing_dir = "/home/mmv/Documents/3.datasets/VOCdevkit"

    annotations_dir = "/home/mmv/Documents/3.datasets/openlogo/Annotations/"

    ####### Model params
    batch_size = 16
    epochs = 160
    lr = 0.0001
    decay_lrs = {60: 0.00001, 90: 0.000001}
    weight_decay = 0.0005
    momentum = 0.9

    num_workers = 8


    # dark 416,416
    im_w = 416
    im_h = 416

    continue_training = False

    ####### Model save/load path
    best_model_path = "testmodel.pt"
    model_path = "testmodel_last.pt"

    ####### YOLO anchors from: https://github.com/uvipen/Yolo-v2-pytorch
    anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
    S = 14
    B = len(anchors)

    thresh = .6

    # testing
    conf_thresh = 0.005
    nms_thresh = .45

    ####### Loss
    object_scale = 5
    noobject_scale = 1
    class_scale = 1
    coord_scale = 1

    debug = False
class Config:

    ####### Dataset
    dataset_dir = "path_to_VOC_dataset"

    ####### Model params
    batch_size = 16
    epochs = 160
    lr = 0.0001
    decay_lrs = {60: 0.00001, 90: 0.000001}
    weight_decay = 0.0005
    momentum = 0.9

    num_workers = 8

    im_w = 416
    im_h = 416

    continue_training = False

    thresh = .6

    jitter = 0.3

    saturation = 1.5
    exposure = 1.5
    hue = .1

    strides = 32

    ####### Model save/load path
    best_model_path = "testmodel.pt"
    model_path = "testmodel_last.pt"

    ####### YOLO anchors from: https://github.com/uvipen/Yolo-v2-pytorch
    anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]

    # testing
    conf_thresh = 0.5
    nms_thresh = .45

    mAP_path = "/home/mmv/Documents/2.projects/Object-Detection-Metrics-master/"  # path to mAP repository

    ####### Loss
    object_scale = 5
    noobject_scale = 1
    class_scale = 1
    coord_scale = 1


class Config:

    ####### Dataset
    training_dir = "/home/camilo/Documents/3.datasets"
    testing_dir = "/home/mmv/Documents/3.datasets/openlogo/preproc/3/testing/"

    annotations_dir = "/home/mmv/Documents/3.datasets/openlogo/Annotations/"

    ####### Model params
    train_batch_size = 32
    train_number_epochs = 1000
    lrate = 0.0005

    # dark 416,416
    im_w = 448
    im_h = 448

    continue_training = False

    ####### Model save/load path
    best_model_path = "testmodel.pt"
    model_path = "testmodel_last.pt"

    ####### YOLO anchors from: https://github.com/uvipen/Yolo-v2-pytorch
    anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
    S = 14
    B = len(anchors)

    iou_thresh = 0.5

    ####### Loss
    alpha_coord = 0.5
    alpha_conf = 100
    alpha_noobj = 5
    alpha_cls = 5
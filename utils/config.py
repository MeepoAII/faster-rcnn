from pprint import pprint

class Config():

    voc_data_dir = '/home/507/myfiles/data/voc/VOCdevkit/VOC2007'
    min_size = 600
    max_size = 1000
    num_workers = 8
    test_num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in original paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1
    lr = 1e-3

    # visualization
    env = 'faster-rcnn'
    port = 8097
    plot_every = 40

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 14

    use_adam = False
    use_chainer = False
    use_drop = False
    # debug
    debug_file = '/tmp/debugf'

    test_num = 1000
    # model
    load_path = None

    caffe_pretrain = False
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('Unknown Option: ', "--%s" % k)
            setattr(self, k, v)


    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


# opt = Config()
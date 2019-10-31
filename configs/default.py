from yacs.config import CfgNode as CN

_C = CN()
_C.NAME = "default"
_C.DATASET = "Flickr8k"
_C.IMG_PATH = "D:/Flickr8k/images/"
_C.ANNOTATIONS_PATH = "D:/Flickr8k/annotations/"
_C.ATTENTION = True

_C.ENCODER = CN()
_C.ENCODER.MODEL = 'VGG16'
_C.ENCODER.BATCH_SIZE = 16

_C.DECODER = CN()
_C.DECODER.BATCH_SIZE = 32
_C.DECODER.EPOCHS = 5
_C.DECODER.INITIAL_STATE_SIZE = 512
_C.DECODER.EMBEDDING_OUT_SIZE = 512
_C.DECODER.NUM_RNN_LAYERS = 2
_C.DECODER.BATCH_NORM = True
_C.DECODER.DROPOUT = True
# if false LSTM is used
_C.DECODER.GRU = True
_C.DECODER.ATTN_TYPE = "bahdanau"
_C.DECODER.MAX_LEN = 30


def update_config(cfg, filename):
    cfg.defrost()
    cfg.merge_from_file(filename)
    cfg.freeze()

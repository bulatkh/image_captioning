from yacs.config import CfgNode as CN

_C = CN()
_C.NAME = "default"
_C.DATASET = "Flickr8k"
_C.ATTENTION = False


_C.PATH = CN()
_C.PATH.IMG_PATH = "D:/Flickr8k/images/"
_C.PATH.ANNOTATIONS_PATH = "D:/Flickr8k/annotations/"
_C.PATH.FEATURES_PATH = "./cnn_features/"
_C.PATH.MODELS_ARCHITECTURE_PATH = "./models/"
_C.PATH.WEIGHTS_PATH = "./weights/"
_C.PATH.CALLBACKS_PATH = "./callbacks/"
_C.PATH.VOCABULARY_PATH = "./vocabulary/"

_C.VOCABULARY = CN()
_C.VOCABULARY.WORD_TO_ID = "word_to_id.pickle"
_C.VOCABULARY.ID_TO_WORD = "id_to_word.pickle"
_C.VOCABULARY.COUNT = "word_counter.pickle"

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
_C.DECODER.OPTIMIZER = 'RMSprop'
_C.DECODER.LR = 0.001
_C.DECODER.DECAY = 0.00000001
_C.DECODER.LOSS = "categorical_crossentropy"
_C.DECODER.SAVE_BEST = True
_C.DECODER.MONITOR = "val_loss"
_C.DECODER.FACTOR = 0.5
_C.DECODER.PATIENCE = 2
_C.DECODER.VERBOSE = 1
_C.DECODER.MIN_LR = 0.0000001
_C.DECODER.VAL_STEPS = 5


def update_config(cfg, filename):
    cfg.defrost()
    cfg.merge_from_file(filename)
    cfg.freeze()

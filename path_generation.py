class PathGenerator(object):
    def __init__(self, gru, dataset, layers, batch_size, batch_norm, drop, attention, attn_type, beam_size=3):
        attn = ''
        dr = ''
        bn = ''
        if gru:
            model = 'GRU'
        else:
            model = 'LSTM'
            if attention:
                attn = '_attn_' + attn_type
        if batch_norm:
            bn = '_bn'
        if drop:
            dr = '_dr'
        beam = '_' + str(beam_size) + 'b'
        self._model_path = './model_files/models/VGG16_{}_{}l{}{}{}.json'.format(
            model,
            layers,
            bn,
            dr,
            attn)
        self._weights_path = './model_files/weights/VGG16_{}_{}_{}l_{}b{}{}{}.hdf5'.format(
            model,
            dataset,
            layers,
            batch_size,
            bn,
            dr,
            attn)
        self._callbacks_path = './model_files/callbacks/VGG16_{}_{}_{}l_{}b{}{}{}.csv'.format(
            model,
            dataset,
            layers,
            batch_size,
            bn,
            dr,
            attn)
        self._captions_path = './model_files/captions/VGG16_{}_{}_{}l_{}b{}{}{}{}.txt'.format(
            model,
            dataset,
            layers,
            batch_size,
            bn,
            dr,
            attn,
            beam)

    def get_model_path(self):
        return self._model_path

    def set_model_path(self, path):
        self._model_path = path

    def get_weights_path(self):
        return self._weights_path

    def set_weights_path(self, path):
        self._weights_path = path

    def get_callbacks_path(self):
        return self._callbacks_path

    def set_callbacks_path(self, path):
        self._callbacks_path = path

    def get_captions_path(self):
        return self._captions_path

    def set_captions_path(self, path):
        self._captions_path = path

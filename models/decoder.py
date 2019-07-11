from keras import backend as K
from keras import Model
from keras.layers import Input, Dense, LSTM, add, Embedding, GRU, Dropout, Multiply, Dot, Lambda, BatchNormalization, \
    RepeatVector, concatenate
from keras.models import model_from_json

import numpy as np


class Decoder:
    """
    Class which is used to build decoder model (keras.Model) instance of a specific architecture.
    """
    def __init__(self, state_size, emb_size, layers, gru, batch_norm, dropout, attn, attn_type, transfer_values,
                 vocab, max_len=30):
        self.state_size = state_size
        self.embedding_size = emb_size
        self.layers = layers
        self.gru = gru
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.attn = attn
        self.attn_type = attn_type
        self.transfer_values = transfer_values
        self.max_len = max_len
        self.vocab = vocab
        if attn:
            self.encoder_input = Input(shape=(transfer_values.shape[1], transfer_values.shape[2]), name='encoder_input')
        else:
            self.encoder_input = Input(shape=(transfer_values.shape[1],), name='encoder_input')
            self.encoder_reduction = Dense(self.state_size, activation='relu', name='encoder_reduction')
        if batch_norm:
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()
            self.bn3 = BatchNormalization()
            self.bn4 = BatchNormalization()
        if dropout:
            self.drop1 = Dropout(0.5)
        if self.gru:
            self.decoder_input = Input(shape=(None,), name='decoder_input')
            self.gru1 = GRU(self.state_size, name='GRU1', return_sequences=True)
            self.gru2 = GRU(self.state_size, name='GRU2', return_sequences=True)
            self.gru3 = GRU(self.state_size, name='GRU3', return_sequences=True)
        else:
            self.repeat = RepeatVector(self.max_len)
            self.decoder_input = Input(shape=(self.max_len,), name='decoder_input')
            if attn:
                self.lstm_att = LSTM(self.state_size, return_state=True)
                self.lstm_att2 = LSTM(self.state_size, return_sequences=True)
                self.densor_s = Dense(self.state_size)
                self.densor_feat = Dense(self.state_size)
                self.gating_scalar_func = Dense(self.state_size, activation='sigmoid')
                self.densor2 = Dense(1)
            else:
                self.lstm1 = LSTM(self.state_size, name='LSTM1', return_sequences=True)
                self.lstm2 = LSTM(self.state_size, name='LSTM2', return_sequences=True)
                self.lstm3 = LSTM(self.state_size, name='LSTM3', return_sequences=True)

        self.embedding = Embedding(input_dim=self.vocab.number_of_words, output_dim=self.embedding_size,
                                   mask_zero=True, name='embedding')
        self.decoder_dense = Dense(self.vocab.number_of_words, activation='softmax', name='decoder_output')

    def build_model(self):
        if self.gru:
            decoder_output = self._connect_transfer_values_gru()
        else:
            if self.attn:
                decoder_output = self._connect_transfer_values_lstm_attention()
            else:
                decoder_output = self._connect_transfer_values_lstm()
        decoder_model = Model(inputs=[self.encoder_input, self.decoder_input], outputs=[decoder_output])
        return decoder_model

    def _connect_transfer_values_gru(self):
        """
        Connects extracted image features to sentences and passes to GRU.
        Image features are the initial state of GRU while sentences are the first input words.
        """
        initial_state = self.encoder_reduction(self.encoder_input)
        if self.batch_norm:
            initial_state = self.bn1(initial_state)
        # pass sentences to embedding
        X = self.decoder_input
        X = self.embedding(X)
        if self.dropout:
            X = self.drop1(X)
        X = self.gru1(X, initial_state=initial_state)
        if self.batch_norm:
            X = self.bn2(X)
        if self.layers >= 2:
            X = self.gru2(X, initial_state=initial_state)
            if self.batch_norm:
                X = self.bn3(X)
        if self.layers == 3:
            X = self.gru3(X, initial_state=initial_state)
            if self.batch_norm:
                X = self.bn4(X)
        # pass the outputs of RNNs to final dense layer which returns a one-hot vector for each word
        decoder_output = self.decoder_dense(X)
        return decoder_output

    def _connect_transfer_values_lstm(self):
        """
        Connects extracted image features to sentences and passes to LSTM.
        Concatenated image features and sentences are LSTM inputs.
        """
        features = self.encoder_reduction(self.encoder_input)
        if self.batch_norm:
            features = self.bn1(features)
        features = self.repeat(features)

        X = self.decoder_input
        X = self.embedding(X)
        if self.dropout:
            X = self.drop1(X)

        X = concatenate([features, X])
        print(X.shape)
        X = self.lstm1(X)
        print(X.shape)

        if self.batch_norm:
            X = self.bn2(X)
        if self.layers >= 2:
            X = self.lstm2(X)
            if self.batch_norm:
                X = self.bn3(X)
        if self.layers == 3:
            X = self.lstm3(X)
            if self.batch_norm:
                X = self.bn4(X)

        decoder_output = self.decoder_dense(X)
        return decoder_output

    def _connect_transfer_values_lstm_attention(self):
        """
        Connects the transfer values to words and pass to LSTM with attention.
        """
        print('Initial features shape', self.encoder_input.shape)
        X = self.decoder_input
        X = self.embedding(X)
        print('word-embedding', X.shape)
        if self.dropout:
            X = self.drop1(X)
        print('Initial states')
        s0 = Lambda(lambda x: K.mean(x, axis=1))(self.encoder_input)
        s0 = Dense(self.state_size, activation='relu')(s0)
        s0 = BatchNormalization()(s0)
        s = s0
        print('s initial', s.shape)
        c0 = Lambda(lambda x: K.mean(x, axis=1))(self.encoder_input)
        c0 = Dense(self.state_size, activation='relu')(c0)
        c0 = BatchNormalization()(c0)
        c = c0
        print('c initial', c.shape)
        lstm_att_out = []
        for i in range(self.max_len):
            print('------------------------')
            print('LSTM iteration {}'.format(i))
            if self.attn_type == 'bahdanau':
                context = self._bahdanau_attention(s, i)
            elif self.attn_type == 'scaled_dot':
                context = self._scaled_dot_product_attention(s, i)
            else:
                raise ValueError('No such attention mechanism')
            print('context', context.shape)
            tmp_X = Lambda(lambda x, t: K.expand_dims(x[:, t], axis=1), arguments={'t': i},
                           output_shape=lambda s: (s[0], 1, s[2]))(X)
            print('current word vector', tmp_X.shape)
            concat = concatenate([context, tmp_X])
            print('lstm input: context-word concat', concat.shape)
            s, _, c = self.lstm_att(concat, initial_state=[s, c])
            print('hidden state', s.shape)
            lstm_att_out.append(s)
        out = Lambda(lambda x: K.stack(x, axis=1))(lstm_att_out)
        print('final lstm output shape', X.shape)
        if self.batch_norm:
            out = self.bn2(out)
        if self.layers == 2:
            out = self.lstm_att2(out, initial_state=[s0, c0])
        if self.batch_norm:
            out = self.bn3(out)
        decoder_output = self.decoder_dense(out)
        print('output', decoder_output.shape)
        return decoder_output

    def _scaled_dot_product_attention(self, s_prev, i):
        """
        Produces context vector for a given pair of image features and previous hidden state using scaled
        dot-product attention

        :param s_prev: previous state of LSTM
        :param i: the number of LSTM iteration
        """
        print('------------------------')
        print('Attention')
        print('img features', self.encoder_input.shape)
        print('prev state', s_prev.shape)
        s_prev = Lambda(lambda x: K.expand_dims(x, 1))(s_prev)
        dot_prod = Dot(axes=2)([self.encoder_input, s_prev])
        print('dot prod', dot_prod.shape)
        scaled_dot_prod = Lambda(lambda x: x / np.sqrt(512))(dot_prod)
        print('dot prod', dot_prod.shape)
        weights = self.densor2(scaled_dot_prod)
        weights = Lambda(lambda x: K.softmax(x, axis=1), name='weights_{}'.format(i))(weights)
        print('weights', weights.shape)
        context = Dot(axes=1)([weights, self.encoder_input])
        print('context', context.shape)
        print('------------------------')
        return context

    def _bahdanau_attention(self, s_prev, i):
        """
        Produces context vector for a given pair of image features and previous hidden state using
        Bahdanau additive attention

        :param s_prev: previous state of LSTM
        :param i: the number of LSTM iteration
        """
        print('------------------------')
        print('Attention')
        print('img features', self.encoder_input.shape)
        print('prev state', s_prev.shape)
        a_dense = self.densor_feat(self.encoder_input)
        print('a_dense', a_dense.shape)
        s_prev = Lambda(lambda x: K.expand_dims(x, 1))(s_prev)
        s_dense = self.densor_s(s_prev)
        print('s_dense', s_dense.shape)
        sum_dense = add([a_dense, s_dense])
        print('summary', sum_dense.shape)
        concat = Lambda(lambda x: K.tanh(x))(sum_dense)
        print('first_dense', concat.shape)
        weights = self.densor2(concat)
        weights = Lambda(lambda x: K.softmax(x, axis=1), name='weights_{}'.format(i))(weights)
        print('weights', weights.shape)
        context = Dot(axes=1)([weights, self.encoder_input])
        gating_scalar = self.gating_scalar_func(s_prev)
        context = Multiply()([context, gating_scalar])
        print('context', context.shape)
        print('------------------------')
        return context


def load_model(model_path, weights_path):
    """
    Loads a keras Model from json and its weights from hdf5
    :param model_path: path to json file
    :param weights_path: path to weights path
    :return: keras.Model instance
    """
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    decoder_model = model_from_json(loaded_model_json)
    decoder_model.load_weights(weights_path)
    return decoder_model


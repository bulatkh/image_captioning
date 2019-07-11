import numpy as np
from keras.preprocessing.sequence import pad_sequences


def batch_one_hot_encode(batch, number_of_words):
    """
    Applies one-hot encoding to a given batch

    :param batch: batch of tokens
    :param number_of_words: number of words in a vocab
    :return: one-hot encoded batch
    """
    batch_size = batch.shape[0]
    sentence_size = batch.shape[1]

    one_hot_batch = np.zeros((batch_size, sentence_size, number_of_words))

    for i in range(batch_size):
        for j in range(sentence_size):
            one_hot_batch[i, j, batch[i, j]] = 1
    return one_hot_batch


def generate_batch(transfer_values, captions_tokens, number_of_words, gru=True, max_length_lstm=30, batch_size=32):
    """
    Generate a batch of input-output data pairs:
        input_data = {
            transfer_values,
            input_tokens
        }

        output_data = {
            output_tokens
        }

    :param transfer_values: encoded images features
    :param captions_tokens: list with all the captions
    :param number_of_words: number of words in vocab
    :param gru: flag for gru model: gru if true, otherwise lstm
    :param max_length_lstm: maximum length of words for lstm model
    :param batch_size: the number of examples in a batch
    :return: pair of input and output batches
    """
    while True:
        # randomly select indices
        indices = np.random.randint(0, len(transfer_values), size=batch_size)
        captions_batch = []
        # randomly select one caption for each example index
        for ind in indices:
            num_captions = len(captions_tokens[ind])
            selected_caption = [0] * (max_length_lstm + 2)
            while len(selected_caption) > max_length_lstm:
                selected_caption = captions_tokens[ind][np.random.randint(0, num_captions - 1)]
            captions_batch.append(selected_caption)

        if not gru:
            # For lstm we will pad sequence array for a fixed length
            captions_batch_padded = pad_sequences(captions_batch,
                                                  maxlen=max_length_lstm + 1,
                                                  padding='post',
                                                  value=0)
        else:
            # Find the largest caption length and pad the remaining to be the same size
            max_caption_size = max([len(cap) for cap in captions_batch])
            captions_batch_padded = pad_sequences(captions_batch,
                                                  maxlen=max_caption_size,
                                                  padding='post',
                                                  value=0)
        # Input tokens are the initial ones starting from index 1
        # Output tokens are the initial ones shifted to the right
        input_tokens = captions_batch_padded[:, :-1]
        output_tokens = captions_batch_padded[:, 1:]

        output_tokens = batch_one_hot_encode(output_tokens, number_of_words)

        input_transfer_values = transfer_values[indices]

        input_data = {
            'encoder_input': input_transfer_values,
            'decoder_input': input_tokens
        }

        output_data = {
            'decoder_output': output_tokens
        }

        yield (input_data, output_data)

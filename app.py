from flask import Flask, flash, render_template, send_from_directory, request, redirect, url_for
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

import numpy as np
import os
import path_generation
from models import predict, transfer_models, decoder
import tensorflow as tf
import text_processing

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}


global graph
graph = tf.get_default_graph()

batch_size = 32
dataset = 'flickr8k'
initial_state_size = 512
embedding_out_size = 512
number_of_layers = 2
max_len = 30
batch_norm = True
dropout = True
gru = False
attn = True
attn_type = 'bahdanau'

path_gen = path_generation.PathGenerator(gru, dataset, number_of_layers, batch_size, batch_norm, dropout, attn,
                                         attn_type)
path_checkpoint = path_gen.get_weights_path()
model_path = path_gen.get_model_path()
vgg_model, vgg_image_size = transfer_models.vgg_model(attn)
decoder_model = decoder.load_model(model_path, path_checkpoint)

vocab = text_processing.Vocabulary()
vocab.load_vocabulary(
    './vocabulary/word_to_id.pickle',
    './vocabulary/id_to_word.pickle',
    './vocabulary/word_counter.pickle')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config.from_object('config.Config')
Bootstrap(app)


@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    checkbox = request.form.get('test')
    beam_size = request.form.get('beam', type=int)

    if not beam_size:
        flash('Beam size form should not be empty')
        return redirect(url_for('index'))

    if not os.path.exists(app.config['WEIGHTS_FOLDER']):
        os.mkdir(app.config['WEIGHTS_FOLDER'])

    if not checkbox:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('index'))

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('index'))

        if file and allowed_file(file.filename):
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.mkdir(app.config['UPLOAD_FOLDER'])

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        else:
            flash('Inappropriate file extension. Try .jpg or .jpeg files.')
            return redirect(url_for('index'))
    else:
        filenames = os.listdir(app.config['TEST_FOLDER'])
        ind = np.random.randint(len(filenames))
        filename = filenames[ind]
        file_path = os.path.join(app.config['TEST_FOLDER'], filenames[ind])

    with graph.as_default():
        captions, _, weights = predict.generate_caption(
            file_path,
            vgg_image_size,
            decoder_model,
            vgg_model,
            vocab,
            transfer_values=False,
            beam_size=beam_size,
            attn=attn,
            get_weights=True
        )
    captions_str = predict.transform_captions(captions)
    weights_filename = filename.rsplit('.', 1)[0] + '_weights_' + str(beam_size) + '.png'
    weights_img_path = os.path.join(app.config['WEIGHTS_FOLDER'], weights_filename)
    predict.get_weights_plot(captions[0], weights, file_path, 'save', weights_img_path)
    return render_template('index.html', imagesource=file_path, captions=captions_str,
                           weights_image=weights_img_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/test_images/<filename>')
def test_file(filename):
    return send_from_directory(app.config['TEST_FOLDER'], filename)


@app.route('/weights_img/<filename>')
def test_weights_file(filename):
    return send_from_directory(app.config['WEIGHTS_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)

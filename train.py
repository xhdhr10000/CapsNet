from __future__ import print_function
from keras import backend as K
from keras.layers import Layer
from keras import activations, optimizers
from keras import utils
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import logging
import os
import numpy as np

import args
from capsule import Capsule
from capsnet import CapsNet
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
FLAGS = args.get()

# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)

def load_weights(model):
    epoch = 0
    path_info = os.path.join(FLAGS.model_dir, 'info')
    if os.path.isfile(path_info):
        f = open(path_info)
        filename = f.readline().strip()
        f.close()
        path = os.path.join(FLAGS.model_dir, filename)
        if os.path.isfile(path):
            logging.info('Loading %s' % filename)
            model.load_weights(path)
            epoch = int(filename.split('_')[1].split('.')[0])
    return epoch

def named_logs(model, logs, step):
  result = { 'batch': step, 'size': FLAGS.batch_size }
  for l in zip(model.metrics_names, logs):
    result[l[0]] = l[1]
  return result

def main(not_parsed_args):
    # we use a margin loss
    model = CapsNet()
    last_epoch = load_weights(model)
    model.compile(loss=margin_loss, optimizer=optimizers.Adam(FLAGS.lr), metrics=['accuracy'])
    model.summary()

    dataset = Dataset(FLAGS.dataset, FLAGS.batch_size)
    tensorboard = TensorBoard(log_dir='./tf_logs', batch_size=FLAGS.batch_size, write_graph=False, write_grads=True, write_images=True, update_freq='batch')
    tensorboard.set_model(model)

    for epoch in range(last_epoch, FLAGS.epochs):
        logging.info('Epoch %d' % epoch)
        model.fit_generator(generator=dataset,
            epochs=1,
            steps_per_epoch=len(dataset)/FLAGS.batch_size,
            verbose=1,
            validation_data=dataset.eval_dataset,
            validation_steps=len(dataset.eval_dataset)/FLAGS.batch_size)

        logging.info('Saving model')
        filename = 'model_%d.h5' % (epoch)
        path = os.path.join(FLAGS.model_dir, filename)
        path_info = os.path.join(FLAGS.model_dir, 'info')
        model.save_weights(path)
        f = open(path_info, 'w')
        f.write(filename)
        f.close()

if __name__ == '__main__':
    tf.app.run()

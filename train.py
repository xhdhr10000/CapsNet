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
    step = 0
    path_info = os.path.join(FLAGS.model_dir, 'info')
    if os.path.isfile(path_info):
        f = open(path_info)
        filename = f.readline().strip()
        f.close()
        path = os.path.join(FLAGS.model_dir, filename)
        if os.path.isfile(path):
            logging.info('Loading %s' % filename)
            model.load_weights(path)
            epoch = int(filename.split('_')[1])
            step = int(filename.split('.')[0].split('_')[2])
    return epoch, step

def named_logs(model, logs, step):
  result = { 'batch': step, 'size': FLAGS.batch_size }
  for l in zip(model.metrics_names, logs):
    result[l[0]] = l[1]
  return result

def main(not_parsed_args):
    # we use a margin loss
    model = CapsNet()
    last_epoch, last_step = load_weights(model)
    model.compile(loss=margin_loss, optimizer=optimizers.Adam(FLAGS.lr), metrics=['accuracy'])
    model.summary()

    dataset = Dataset(FLAGS.dataset)
    tensorboard = TensorBoard(log_dir='./tf_logs', batch_size=FLAGS.batch_size, write_graph=False, write_grads=True, write_images=True, update_freq='batch')
    tensorboard.set_model(model)

    for epoch in range(last_epoch, FLAGS.epochs):
        tensorboard.on_epoch_begin(epoch)
        for step in range(last_step+1, dataset.count // FLAGS.batch_size):
            tensorboard.on_batch_begin(step)
            x_train, y_train = dataset.load_image(FLAGS.batch_size)
            loss = model.train_on_batch(x_train, y_train)
            logging.info('Epoch %d step %d: loss %.6f accuracy %.6f' % (epoch, step, loss[0], loss[1]))
            tensorboard.on_batch_end(step, named_logs(model, loss, step))

            # if FLAGS.dataset_val and step > 0 and step % FLAGS.val_interval == 0 or step == dataset.count // FLAGS.batch_size - 1:
            #     logging.info('Validation start')
            #     val_loss = 0
            #     val_psnr = 0
            #     for _ in range(len(val_set)):
            #         x_val, y_val = val_set.batch(1)
            #         score = model.test_on_batch(x_val, y_val)
            #         val_loss += score[0]
            #         val_psnr += score[1]
            #     val_loss /= len(val_set)
            #     val_psnr /= len(val_set)
            #     logging.info('Validation average loss %f psnr %f' % (val_loss, val_psnr))

            if step > 0 and step % FLAGS.save_interval == 0:
                logging.info('Saving model')
                filename = 'model_%d_%d.h5' % (epoch, step)
                path = os.path.join(FLAGS.model_dir, filename)
                path_info = os.path.join(FLAGS.model_dir, 'info')
                model.save_weights(path)
                f = open(path_info, 'w')
                f.write(filename)
                f.close()
        last_step = -1

if __name__ == '__main__':
    tf.app.run()
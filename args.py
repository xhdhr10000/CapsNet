import tensorflow as tf
import sys
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 100, 'Training epochs')
flags.DEFINE_integer('batch_size', 16, 'Batch size for training')
flags.DEFINE_float('lr', 0.0001, 'Learning rate')
flags.DEFINE_string('dataset', 'dataset', 'Dataset for training')
flags.DEFINE_string('dataset_val', None, 'Dataset for validation')
flags.DEFINE_string('model_dir', 'models', 'Model save directory')
flags.DEFINE_integer('val_interval', 200, 'Validation interval')
flags.DEFINE_integer('save_interval', 200, 'Model save interval')

def get():
    print("Python Interpreter version:%s" % sys.version[:3])
    print("tensorflow version:%s" % tf.__version__)
    print("numpy version:%s" % np.__version__)

    # check which library you are using
    # np.show_config()
    return FLAGS

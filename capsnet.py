from keras import backend as K
from keras.layers import Input, Conv2D, AveragePooling2D, Reshape, Lambda, Dropout
from keras.models import Model
from capsule import Capsule

def CapsNet():
    # A common Conv2D model
    input_image = Input(shape=(None, None, 3))
    x = Conv2D(64, (3, 3), activation='relu')(input_image)
    x = Dropout(0.5)(x)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)

    """now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
    then connect a Capsule layer.

    the output of final model is the lengths of 120 Capsule, whose dim=16.

    the length of Capsule is the proba,
    so the problem becomes a 120 two-classification problem.
    """

    x = Reshape((-1, 128))(x)
    capsule = Capsule(120, 16, 3, True)(x)
    output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
    return Model(inputs=input_image, outputs=output)

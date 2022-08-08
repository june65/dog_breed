import keras

import tensorflow_hub as hub

model = keras.models.load_model('./dog_breed_model')

model.save('./h5_model/dog_breed_model.h5')

import cv2 
from skimage.feature import hog
import argparse
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model 
from tensorflow.keras.applications import xception
import numpy as np
from tensorflow import keras 



# !pip install git+https://github.com/rcmalli/keras-vggface.git
# !pip install keras_applications --no-deps
filename =r'C:\Users\Admin\AppData\Local\Programs\Python\Python37\lib\site-packages\keras_vggface\models.py'
text = open(filename).read()
open(filename, "w+").write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))
import tensorflow as tf
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

vggface = VGGFace(model='resnet50') # or VGGFace() as default
class VGG16_FE:

    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img ):
        img = img.resize( (224, 224))
        img=img.convert('RGB')
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        feature = self.model.predict(img_data)[0]

        return feature/np.linalg.norm(feature)

class Xception_FE:
    def __init__(self):
        base_model = xception.Xception(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    def extract(self, img):
        img = img.resize((299, 299))
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = xception.preprocess_input(x)
        feature = self.model.predict(x)[0]
        feature = feature / np.linalg.norm(feature)

        return feature
class VGGFACE_FE:
    def __init__(self):
        self.model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    def extract(self, img):
        img = img.resize((224, 224))
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)
        feature = self.model.predict(x)[0]
        feature = feature / np.linalg.norm(feature)

        return feature
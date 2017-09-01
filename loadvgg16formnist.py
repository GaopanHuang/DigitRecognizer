import tensorflow as tf
#config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4

sess = tf.Session(config=config)

from keras import backend as K
K.set_session(sess)


import keras
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from PIL import Image as pilimage
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np

# load json and create model
json_file = open('modelvgg16.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelvgg16.h5")
print("Loaded model from disk")

def resizeimg(arr):
    a = arr.reshape((-1,28,28))
    a = a[:,:,:,np.newaxis]
    x = np.empty(shape=[0,224,224,3])
    for i in range(len(a)):
        img = a[i,:,:,:]
        img = image.array_to_img(img)
        hw_tuple = (224,224)
        img = img.resize(hw_tuple)
        # need to add grayimg to rgb
        img = img.convert("RGB")
        x1 = image.img_to_array(img)
        x1 = x1[np.newaxis,:,:,:]
        x = np.append(x, x1, axis=0)
    return preprocess_input(x)

#samples_test = np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=1)
#x_test = resizeimg(samples_test)                                                                              
#y_test = loaded_model.predict(x_test)


samples_test = np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=1)
f_handle = open('test_result_load.csv', 'a')
batch_size = 280
epoch = 100
for i in range(epoch):
    x_test = resizeimg(samples_test[i*batch_size:(i+1)*batch_size,:])
    print("%d batch loaded" % i)
    y_test = loaded_model.predict(x_test, batch_size=28)
    print("predict done")

    rlt = np.empty(shape=[0,2])
    for j in range(batch_size):
        index = j+i*batch_size+1
        a = np.array([index,y_test[j].argmax()])
        a = a[np.newaxis,:]
        rlt = np.append(rlt, a, axis=0)
    np.savetxt(f_handle,rlt,fmt='%d',delimiter=',')
    print("saved batch result")

f_handle.close()
print("saved result") 

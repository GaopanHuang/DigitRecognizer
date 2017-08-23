import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from PIL import Image as pilimage
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np

train_num = 32000

base_model = VGG16(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 10 classes
predictions = Dense(10, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

#model.summary()

from keras.utils import plot_model
plot_model(model, to_file='model_vgg16formnist.png')

#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
#for layer in base_model.layers:
#    layer.trainable = False
for layer in model.layers[:11]:
   layer.trainable = False
for layer in model.layers[11:]:
   layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

def resizeimg(arr):
    a = arr.reshape((28,28))
    img = pilimage.fromarray(a)
    img = img.convert("RGB")
    img = img.resize((224,224),pilimage.ANTIALIAS)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)                                 
    return preprocess_input(x)

def generate_train(path):
    while 1:
        samples = np.loadtxt(open("train.csv","rb"),delimiter=",",skiprows=1)
        for i in range(train_num):
            # create numpy arrays of input data
            # and labels, from each line in the file
            x = resizeimg(samples[i,1:])
            y = keras.utils.to_categorical(samples[i,0], num_classes=10)
            #yield ({'input_1': x}, {'output': y})
            yield (x, y)
        f.close()

def generate_val(path):
    while 1:
        samples = np.loadtxt(open("train.csv","rb"),delimiter=",",skiprows=train_num+1)
        for i in range(len(samples)):
            # create numpy arrays of input data
            # and labels, from each line in the file
            x = resizeimg(samples[i,1:])
            y = keras.utils.to_categorical(samples[i,0], num_classes=10)
            #yield ({'input_1': x}, {'output': y})
            yield (x, y)
        f.close() 

tensorboard = TensorBoard(log_dir='./logs')

model.fit_generator(generator=generate_train('train.csv'),steps_per_epoch=1000,
    epochs=100,callbacks=[tensorboard], 
    validation_data=generate_val('train.csv'), validation_steps=100)

'''
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
'''
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

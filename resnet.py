import os
from keras.preprocessing import image
from PIL import Image
import math
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from lrFinder import LRFinder
import kerastuner as kt
import json
import keras.backend as K

POPULATE_THRESHOLD = 100

train_path = "C:/Users/beno/PycharmProjects/PokemonClassification/pokemon/dataset/train_images"
val_path = "C:/Users/beno/PycharmProjects/PokemonClassification/pokemon/dataset/validation_images"
root = "C:/Users/beno/PycharmProjects/PokemonClassification/pokemon/dataset"

# fetches with the the absolute path
each_pokemon_folder = [root + "/" + name for name in os.listdir("pokemon/dataset")]

each_pokemon_folder_train = [train_path + "/" + name for name in os.listdir("pokemon/dataset/train_images")]

each_pokemon_folder_validation = [val_path + "/" + name for name in os.listdir("pokemon/dataset/validation_images")]


def folder_len(folder_path):
    return len([name for name in os.listdir(folder_path)])


def fetch_images_from_folder(folder_path):
    return [folder_path + "/" + i for i in os.listdir(folder_path) if os.path.isfile(folder_path + "/" + i)]


# If less then 100 images folder path will be sent to data augmentation
# t -> Threshold for image count ( POPULATE_THRESHOLD )
def should_aug(folder_path, t):
    return True if folder_len(folder_path) < t else False


# Returns the list of absolute paths of the folders with less than "POPULATE_THRESHOLD" (100 for the first case) images.
def folders_to_aug():
    return [i for i in each_pokemon_folder if should_aug(i, POPULATE_THRESHOLD)]


# To see image counts of each folder
def total_counts(path):
    for i in range(len(path)):
        print("Total count of " + path[i].rsplit("/")[-1] + " " + str(
            folder_len(path[i])))


#total_counts(each_pokemon_folder_train)


# Used once
def createTrainFolders():
    for i in each_pokemon_folder:
        os.mkdir(os.path.join(train_path, i.rsplit("/")[-1]))


# Used once
def createValFolders():
    for i in each_pokemon_folder:
        os.mkdir(os.path.join(val_path, i.rsplit("/")[-1]))


# Moves the files to folders respectively with desired split rate...
# If you want to change the split rate just delete train_images and validation_images and make them again (empty)
# DID IT ONCE!!!!!
def splitFolders(folders, SPLIT_RATE):
    # to skip unnecessary folders
    for i in folders:
        if i == "C:/Users/beno/PycharmProjects/PokemonClassification/pokemon/dataset/train_images":
            continue
        elif i == "C:/Users/beno/PycharmProjects/PokemonClassification/pokemon/dataset/validation_images":
            continue
        else:
            print(i)
            images = os.listdir(i)
            length = int(SPLIT_RATE * len(images))
            train_images = images[:length]
            val_images = images[length:]
            print("Length of the images " + str(len(images)))
            print("Length of the train images made by split from the total " + str(len(train_images)))
            print("Length of the validation images made by split from the total " + str(len(val_images)))
            # Train images
            for t in train_images:
                shutil.copy(os.path.join(i, t), os.path.join(train_path, i.rsplit("/")[-1]))
            # Validation images
            for v in val_images:
                shutil.copy(os.path.join(i, v), os.path.join(val_path, i.rsplit("/")[-1]))



# Executed here...
# splitFolders(each_pokemon_folder, 0.8)
# exit()


# Data Augmentation
from PIL import Image
import math
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

dataGen = ImageDataGenerator(rescale=1.0 / 255,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             rotation_range=40,
                             fill_mode='nearest'
                             )



dataGenSimple = ImageDataGenerator(rescale=1.0 / 255)


# Taking copies with Augmentation to the relative files
# Which can be a solution to balance distribution of instances for each class,
# so that I would neither need to calculate precision nor recall.
# This approach might get problematic...
def aug():
    for folders in os.listdir(val_path):
        images_path = os.path.join(val_path, folders)
        img_count = len(os.listdir(images_path))

        if img_count <= 39:
            img_arr = os.listdir(images_path)

            for img in img_arr:

                img_ = image.load_img(os.path.join(images_path, img), target_size=(240, 240))
                img_ = image.img_to_array(img_)
                img_ = img_.reshape(1, 240, 240, 3)

                limit = np.floor(213 / img_count)

                i = 0
                for x in dataGen.flow(img_, batch_size=1, save_to_dir=images_path, save_prefix=folders,
                                      save_format='jpg'):
                    i += 1
                    x = x.reshape(240, 240, 3)
                    img = Image.fromarray(x, 'RGB')
                    pathii = os.path.join(images_path, 'save.png')
                    img.save(pathii)
                    if i >= limit:
                        break


# to how many images you want to reduce to...
def reduce_dataset(reduce_to):
    for folders in os.listdir(val_path):
        for images in os.listdir(os.path.join(val_path, folders))[reduce_to:]:
            os.remove(os.path.join(os.path.join(val_path, folders), images))


print("****************** TRAIN FILES ******************")
total_counts(each_pokemon_folder_train)
print("-------------------------------------------------")
print("****************** VALID FILES ******************")
total_counts(each_pokemon_folder_validation)

exit()

BATCH_SIZE = 32

train_generator = dataGenSimple.flow_from_directory(
    'pokemon/dataset/train_images',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(240, 240),
)

val_generator = dataGenSimple.flow_from_directory(

    directory='pokemon/dataset/validation_images',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(240, 240),
)

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

from keras.models import Sequential
from keras.layers import *

#
# for layer in model.layers:
#     layer.trainable=False

from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras import Model

from keras.optimizers import Adam
from keras.optimizers import SGD



def ResNet():
    model = ResNet50(include_top=False, weights='imagenet', input_shape=(240, 240, 3))

    # Freezing all the layers except conv5 block
    # for layer in model.layers:
    #     if not str(layer.name).startswith("conv5"):
    #         layer.trainable = False

    # Freezing only the first layer since it has some basic features. ( lines, curves etc... )
    # for layer in model.layers:
    #     if str(layer.name).startswith("conv1"):
    #         layer.trainable = False


    model.trainable = True



    # Freezing all layers
    # for layer in model.layers:
    #     layer.trainable = False



    # for layer in model.layers:
    #     if str(layer.name).startswith("conv2") or str(layer.name).startswith("conv3") or \
    #             str(layer.name).startswith("conv4") or str(layer.name).startswith("conv5"):
    #         layer.trainable = True

    # Setting all BN layers as trainable
    for layer in model.layers:
        if "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = True

    for layer in model.layers:
        if layer.trainable == True:
            print(layer.name)

    for layer in model.layers:
        if layer.trainable == False:
            print("Frozen LAYER -> " ,layer.name)


    # Adding top here
    layer1 = GlobalAveragePooling2D()(model.output)
    layer2 = Dropout(0.5)(layer1)
    layer3 = Dense(400, activation='relu')(layer2)
    layer4 = Dense(200, activation='relu')(layer3)
    layer_out = Dense(149, activation='softmax')(layer4)


    opt = Adam(lr=1e-5)
    model_new = Model(inputs=model.input, outputs=layer_out)
    model_new.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # for layers in model_new.layers:
    #     if layers.trainable == True:
    #         print(layers.name)

    return model_new


# Freezing all the layers except conv5 block
# for layer in model.layers:
#     if not str(layer.name).startswith("conv5"):
#         layer.trainable = False

# Unfreezing the BN layers given that it gives the means and variances of pre-trained image-net
# for layer in model.layers:
#     if "BatchNormalization" in layer.__class__.__name__:
#         layer.trainable = True

# Checking unfrozen layers

# for layer in model.layers:
#     if layer.trainable == True:
#         print(layer.name)

adam_optimizer = Adam(lr=1e-4)


def someRandomModel():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(240, 240, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(149, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def AlexNet(input_shape):
    X_input = Input(input_shape)

    X = Conv2D(96, (11, 11), strides=4, name="conv0")(X_input)
    X = BatchNormalization(axis=3, name="bn0")(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2, name='max0')(X)

    X = Conv2D(256, (5, 5), padding='same', name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2, name='max1')(X)

    X = Conv2D(384, (3, 3), padding='same', name='conv2')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)

    X = Conv2D(384, (3, 3), padding='same', name='conv3')(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('relu')(X)

    X = Conv2D(256, (3, 3), padding='same', name='conv4')(X)
    X = BatchNormalization(axis=3, name='bn4')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2, name='max2')(X)

    X = Flatten()(X)

    X = Dense(400, activation='relu', name="fc0")(X)

    X = Dense(400, activation='relu', name='fc1')(X)

    X = Dense(149, activation='softmax', name='fc2')(X)

    sgd_optimizer = SGD(lr=0.01, clipvalue=0.5, momentum=0.9)
    model = Model(inputs=X_input, outputs=X, name='AlexNet')
    model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


from keras import regularizers


def add_top(hp):
    inception_model = tf.keras.applications.InceptionV3(include_top=False, weights=None, input_shape=(240, 240, 3))
    layer1 = GlobalAveragePooling2D()(inception_model.output)
    layer2 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(layer1)
    layer3 = Dense(hp.Int('hidden_size',min_value=512,max_value=1024,step=256), activation='relu', kernel_regularizer=regularizers.l2(0.001))(layer2)
    # layer4 = Dropout(0.3)(layer3)
    layer_out = Dense(149, activation='softmax')(layer3)
    inception = Model(inputs=inception_model.input, outputs=layer_out)
    inception.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # inception.summary()
    return inception



from kerastuner.tuners import RandomSearch

tuner = kt.Hyperband(
    add_top,
    objective='val_accuracy',
    max_epochs=5,
    hyperband_iterations=2,
)

tuner.search(
    train_generator,
    validation_data = val_generator,
    epochs=5,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=1)]
)

best_model = tuner.get_best_models(1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
print(best_model)
print(best_hyperparameters)

from keras.models import load_model

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', verbose=1, patience=4),
    tf.keras.callbacks.ModelCheckpoint(filepath="histories/boş.h5", monitor='val_accuracy',
                                       save_best_only=True),
    tf.keras.callbacks.CSVLogger('histories/boş.log', separator=",", append=False),
]

csv_logger = tf.keras.callbacks.CSVLogger('histories/boş.log')

res = ResNet()


hist = res.fit(train_generator, epochs=100, validation_data=val_generator,
                     steps_per_epoch=train_generator.samples // BATCH_SIZE,
                     validation_steps=val_generator.samples // BATCH_SIZE, callbacks=my_callbacks)



# a = load_model('models/InceptionCheckPointsValAcc0.4601Adam.h5')
# print(len(val_generator.classes))
# a.summary()
# exit()
# a.evaluate(x=val_generator,y=None)
# print(train_generator.class_indices)
# a.evaluate(val_generator)

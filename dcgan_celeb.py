from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.core import Flatten
from keras.optimizers import Adam, SGD
from keras.datasets import mnist
from keras.layers import Input, merge
from keras.models import Model
import numpy as np
from PIL import Image
import argparse
import math
from utils import load_data
from utils import celebA_iter
import keras.backend as K
K.set_image_dim_ordering('tf')

np.random.seed(1337)

def generator_model():
    model = Sequential()
    model.add(Conv2DTranspose(1024, (4, 4),
              strides=(1, 1),
              padding='valid',
              input_shape=(1, 1, 100)))
    model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(512, (4, 4),
              strides=(2, 2),
              padding='same',
              input_shape=(4, 4, 1024)))
    model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(256, (4, 4),
              strides=(2, 2),
              padding='same',
              input_shape=(8, 8, 512)))
    model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4, 4),
              strides=(2, 2),
              padding='same',
              input_shape=(16, 16, 256)))
    model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(3, (4, 4),
              strides=(2, 2),
              padding='same',
              input_shape=(32, 32, 128)))
    model.add(Activation('tanh'))

    g_optim = Adam(lr=0.00015, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=g_optim )
    return model


def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (4, 4),
            strides = (2, 2),
            padding='same',
            input_shape=(64, 64, 3))
    )
    model.add(LeakyReLU(alpha=0.2))
    model.add(
            Conv2D(128, (4, 4),
            strides = (2, 2),
            padding='same',
            input_shape=(32, 32, 64))
    )
    model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(
            Conv2D(256, (4, 4),
            strides = (2, 2),
            padding='same',
            input_shape=(16, 16, 128))
    )
    model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(
            Conv2D(512, (4, 4),
            strides = (2, 2),
            padding='same',
            input_shape=(8, 8, 256))
    )
    model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    d_optim = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=d_optim)
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height*shape[0], width*shape[1],shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = \
            img
    return image


def train(BATCH_SIZE):

    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    d = discriminator_model()
    g = generator_model()
    d.trainable = False
    opt = Adam(lr=0.00015, beta_1=0.5)
    d_on_g = generator_containing_discriminator(g, d)
    d_on_g.compile(loss='binary_crossentropy', optimizer=opt)
    d_on_g.summary()
    for epoch in range(100):
        i = 0
        print("Epoch is {}".format(epoch))
        for true_images in celebA_iter():
            noise = np.random.normal(0, 1, size=(BATCH_SIZE,)+ (1,1,100))
            generated_images = g.predict(noise)
            if i % 20 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    "./images_celebA/"+str(epoch)+"_"+str(i)+".png")
            X = np.concatenate((true_images, generated_images))
            real_data_Y = np.ones(BATCH_SIZE) - np.random.random_sample(BATCH_SIZE)*0.2
            fake_data_Y = np.random.random_sample(BATCH_SIZE)*0.2
            y = np.concatenate((real_data_Y,fake_data_Y))
            d.trainable = True
            g.trainable = False
            dis_metrics_real = d.train_on_batch(true_images,real_data_Y)   #training seperately on real
            dis_metrics_fake = d.train_on_batch(generated_images,fake_data_Y)   #training seperately on fake
            print("epoch: %d, batch: %d Disc: real loss: %f fake loss: %f" % (epoch, i, dis_metrics_real, dis_metrics_fake))
            #d_loss = d.train_on_batch(X, y)
            #print("epoch : {} batch : {} d_loss : {}".format(epoch, i, d_loss))
            #one batch not ok
            g.trainable = True
            noise = np.random.normal(0, 1, size=(BATCH_SIZE,)+ (1,1,100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, real_data_Y)
            print("epoch : {} batch : {} g_loss : {}".format(epoch, i, g_loss))
            i = i+1
        g.save_weights('generator_cele')
        d.save_weights('discriminator_cele')


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="Adam")
    g.load_weights('generator_cele')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise =  np.random.normal(0, 1, size=(BATCH_SIZE,)+ (1,1,100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:4], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, :] = generated_images[idx, :, :, :]
        image = combine_images(nice_images)
    else:
        noise = np.random.normal(0, 1, size=(batch_size,)+ (1,1,100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

def test():
    i=0
    for images in celebA_iter():
        i=i+1


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
    elif args.mode == "test":
        test()

#!/usr/bin/env python3
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import csv
import random
import h5py
import glob
import imageio
import math

import tifffile

import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa

import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image

from skimage import transform
from skimage import exposure
from skimage import color
from skimage.io import imread
from skimage.io import imshow
from skimage.draw import polygon
from skimage.exposure import equalize_hist

from keras import backend as K

from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import ReLU
from keras.layers import MaxPooling2D
from keras.layers import Conv2DTranspose
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Reshape

from keras.utils import Sequence

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import Callback

from keras.models import Model

# from: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
def weighted_categorical_crossentropy(weights):

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

def dice_coefficient(y_true, y_pred):

    eps = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps)

# DataLoader
class SparseDataGenerator(Sequence):

    def __init__(self,
                 data_dir,
                 batch_size=32,
                 shape=(256,256,1),
                 dense=False,
                 augment=True):

        self.dense = dense
        self.augment = augment

        # Training Directory
        img_list = glob.glob(os.path.join(data_dir, 'images/*.png'))
        img_list.sort()

        # Mask directory contains three sub-directories each containing
        # binary mask corresponding to foreground, background and None
        if self.dense:
            msk_list = glob.glob(os.path.join(data_dir, 'masks/*.png'))
            msk_list.sort()

            self.data_list = list(zip(img_list,
                                      msk_list))
        else:
            msk_bg_list = glob.glob(os.path.join(data_dir, 'masks/background/*.png'))
            msk_fg_list = glob.glob(os.path.join(data_dir, 'masks/foreground/*.png'))

            msk_bg_list.sort()
            msk_fg_list.sort()

            self.data_list = list(zip(img_list,
                                      msk_bg_list,
                                      msk_fg_list))

        self.on_epoch_end()
        self.batch_size = batch_size
        self.shape = shape

    def __len__(self):
        if self.augment:
            return 2 * len(self.data_list) // self.batch_size
        else:
            return len(self.data_list) // self.batch_size

    def __getitem__(self, idx):

        # Instantiate batches
        img_batch = np.zeros((self.batch_size,
                              self.shape[0],
                              self.shape[1],
                              self.shape[2]))
        msk_batch = np.zeros((self.batch_size,
                              self.shape[0],
                              self.shape[1],
                              3))

        # If densely annotated, background layer is 1-foreground and None class is 1-bg-fg in all cases
        k = idx % (len(self.data_list) // self.batch_size)
        for i in range(k * self.batch_size , k * self.batch_size + self.batch_size):

            j = i % self.batch_size

            img_batch[j, :, :, 0] = self._imread(self.data_list[i][0],
                                                 (self.shape[0], self.shape[1]),
                                                 'float')
            if self.dense:
                msk_batch[j, :, :, 1] = self._imread(self.data_list[i][1],
                                                     (self.shape[0], self.shape[1]),
                                                     'float',
                                                     binary=True)
                msk_batch[j, :, :, 0] = np.ones((self.shape[0], self.shape[1])) - msk_batch[j, :, :, 1]
            else:
                msk_batch[j, :, :, 0] = self._imread(self.data_list[i][1],
                                                     (self.shape[0], self.shape[1]),
                                                     'float',
                                                     binary=True)
                msk_batch[j, :, :, 1] = self._imread(self.data_list[i][2],
                                                     (self.shape[0], self.shape[1]),
                                                     'float',
                                                     binary=True)

            msk_batch[j, :, :, 2] = np.ones((self.shape[0], self.shape[1])) - msk_batch[j, :, :, 1] - msk_batch[j, :, :, 0]

        if self.augment:

            # Roughly augment 50% of batches
            rand = np.random.random()

            if rand > 0.5:
                return img_batch, msk_batch

            else:
                seq = iaa.Sequential([
                    iaa.Fliplr(0.5), # horizontal flips
                    iaa.Flipud(0.5), # vertical flips
                    iaa.Crop(percent=(0, 0.1)), # random crops
                    iaa.Sometimes(
                        0.5,
                        iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
                    iaa.LinearContrast((0.75, 1.5)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
                    iaa.Multiply((0.8, 1.2)),
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-25, 25),
                        shear=(-8, 8)
                    )
                ], random_order=True) # apply augmenters in random order

                # For augmentation purposes, collapse segmentation masks from 4D to 3D
                msk_batch_temp = np.zeros((msk_batch.shape[0],
                                           msk_batch.shape[1],
                                           msk_batch.shape[2]), dtype=np.int32)
                msk_batch_temp[np.where(msk_batch[:,:,:,0] == 1)] = 0
                msk_batch_temp[np.where(msk_batch[:,:,:,1] == 1)] = 1
                msk_batch_temp[np.where(msk_batch[:,:,:,2] == 1)] = 2
                msk_batch_temp = np.expand_dims(msk_batch_temp, axis=3)

                # msk_batch_temp = SegmentationMapsOnImage(msk_batch_temp, shape=(msk_batch.shape[1], msk_batch.shape[2], 1))
                # Returns a list of images --> Must repopulate an array
                img_aug_list, msk_aug_list = seq(images=(img_batch*255).astype(np.uint8),
                                                 segmentation_maps=msk_batch_temp)

                # Repopulate img array
                img_batch_aug = np.zeros(img_batch.shape)
                for i in range(len(img_aug_list)):
                    img_batch_aug[i] = img_aug_list[i]

                # Re-expand from collapsed mask to one-hot-encoded mask
                msk_batch_aug = np.zeros((msk_batch.shape[0],
                                          msk_batch.shape[1],
                                          msk_batch.shape[2],
                                          3))

                # Assign entire collapsed mask to one channel and then set all but desired value to zero
                for i, msk in enumerate(msk_aug_list):
                    msk = np.squeeze(msk)
                    msk_temp_zero = np.zeros(msk.shape)
                    msk_temp_zero[np.where(msk == 0)] = 1
                    msk_temp_one = np.zeros(msk.shape)
                    msk_temp_one[np.where(msk == 1)] = 1
                    msk_temp_two = np.zeros(msk.shape)
                    msk_temp_two[np.where(msk == 2)] = 1
                    msk_batch_aug[i,:,:,0] = msk_temp_zero
                    msk_batch_aug[i,:,:,1] = msk_temp_one
                    msk_batch_aug[i,:,:,2] = msk_temp_two

                img_batch_aug = (img_batch_aug/255).astype(np.float64)

                return img_batch_aug, msk_batch_aug

        else:
            return img_batch, msk_batch


    def on_epoch_end(self):

        random.shuffle(self.data_list)

    def batch_weights(self):
        # Compute from sample of 100 masks --> must ensure that None class is not counted...
        # Weight of class zero is all zero masks added divided by all
        return (1, 3, 0)

    def _imread(self, img_path, size, dtype, binary=False):

        try:
            img = imageio.imread(img_path, pilmode='L').astype(np.float)

            resized_img = transform.resize(img, size, mode='constant', preserve_range=True)
            out_img = exposure.rescale_intensity(resized_img, out_range=dtype)

            if binary:
                return np.squeeze(out_img) > 0
            else:
                return color.rgb2gray(out_img)
        except ValueError:
            print('Corrupted filename: ', img_path)
            return np.zeros(size)

class SampleImageCallback(Callback):

    def __init__(self, model, sample_data, model_path, save=False):
        self.model = model
        self.sample_data = sample_data
        self.model_path = model_path
        self.save = save

    def on_epoch_end(self, epoch, logs={}):

        sample_predict = self.model.predict_on_batch(self.sample_data)

        f=plt.figure(figsize=(16,8))
        plt.subplot(1,2,1)
        plt.imshow(self.sample_data[0,:,:,0], interpolation='nearest', cmap='gray')
        plt.title('Sample source')
        plt.axis('off');

        plt.subplot(1,2,2)
        plt.imshow(sample_predict[0,:,:,1], interpolation='nearest', cmap='magma')
        plt.title('Predicted target')
        plt.axis('off');

        if self.save:
            plt.savefig(self.model_path + '/epoch_' + str(epoch+1) + '.png')
        else:
            plt.show()

class SparseUnet:
    def __init__(self, shape=(256,256,1)):

        self.shape = shape

        input_img = Input(self.shape, name='img')

        self.model = self.unet_2D(input_img)

    def down_block_2D(self, input_tensor, filters):

        x = Conv2D(filters=filters, kernel_size=(3,3), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(filters=filters*2, kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    def up_block_2D(self, input_tensor, concat_layer, filters):

        x = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2))(input_tensor)

        x = Concatenate()([x, concat_layer])

        x = Conv2D(filters=filters, kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(filters=filters*2, kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    def unet_2D(self, input_tensor, filters=32):

        d1 = self.down_block_2D(input_tensor, filters=filters)
        p1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(d1)
        d2 = self.down_block_2D(p1, filters=filters*2)
        p2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(d2)
        d3 = self.down_block_2D(p2, filters=filters*4)
        p3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(d3)

        d4 = self.down_block_2D(p3, filters=filters*8)

        u1 = self.up_block_2D(d4, d3, filters=filters*4)
        u2 = self.up_block_2D(u1, d2, filters=filters*2)
        u3 = self.up_block_2D(u2, d1, filters=filters)

        # Returns one-hot-encoded semantic segmentation mask where 0 is bakcground, 1 is mito and 2 is None (weight zero)
        output_tensor = Conv2D(filters=3, kernel_size=(1,1), activation='softmax')(u3)

        return Model(inputs=[input_tensor], outputs=[output_tensor])

    def train(self, train_dir, test_dir, out_dir, epochs=100, batch_size=32, dense=False, log_name='log.csv', model_name='sparse_unet'):

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not os.path.exists(os.path.join(out_dir, 'ckpt')):
            os.makedirs(os.path.join(out_dir, 'ckpt'))

        train_generator = SparseDataGenerator(train_dir,
                                              batch_size=batch_size,
                                              shape=self.shape,
                                              dense=dense)

        val_generator = SparseDataGenerator(test_dir,
                                            batch_size=batch_size,
                                            shape=self.shape,
                                            dense=dense,
                                            augment=False)

        sample_batch = val_generator[0][0]
        sample_img = SampleImageCallback(self.model,
                                         sample_batch,
                                         out_dir,
                                         save=True)

        weight_zero, weight_one, weight_two = train_generator.batch_weights()

        self.model.compile(optimizer='adam',
                           loss=weighted_categorical_crossentropy(np.array([weight_zero, weight_one, weight_two])),
                           metrics=[dice_coefficient])

        self.model.summary()

        # self.model.load_weights(model_name)

        csv_logger = CSVLogger(os.path.join(out_dir, log_name))

        ckpt_name =  'ckpt/' + model_name + '_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.hdf5'

        model_ckpt = ModelCheckpoint(os.path.join(out_dir, ckpt_name),
                                     verbose=1,
                                     save_best_only=False,
                                     save_weights_only=True)

        self.model.fit_generator(generator=train_generator,
                                 validation_data=val_generator,
                                 validation_steps=math.floor(len(val_generator))/batch_size,
                                 epochs=epochs,
                                 shuffle=False,
                                 callbacks=[csv_logger,
                                            model_ckpt,
                                            sample_img])

    def predict(self, input, tile_shape):

        pad_in = np.zeros((math.ceil(input.shape[0] / tile_shape[0]) * tile_shape[0], math.ceil(input.shape[1] / tile_shape[1]) * tile_shape[1]))

        pad_in[:input.shape[0],:input.shape[1]] = input

        pad_out = np.zeros(pad_in.shape)

        for x in range(math.ceil(input.shape[0] / tile_shape[0])):
            for y in range(math.ceil(input.shape[1] / tile_shape[1])):
                x_a = x * tile_shape[0]
                x_b = x * tile_shape[0] + tile_shape[0]

                y_a = y * tile_shape[1]
                y_b = y * tile_shape[1] + tile_shape[1]
                pad_out[x_a:x_b,y_a:y_b] = self.tile_predict(pad_in[x_a:x_b,y_a:y_b])

        return pad_out[:input.shape[0],:input.shape[1]]

    def tile_predict(self, input):

        exp_input = np.zeros((1, self.shape[0], self.shape[1], 1))

        exp_input[0,:,:,0] = exposure.equalize_hist(transform.resize(input, (self.shape[0], self.shape[1]), mode='constant', preserve_range=True))

        return transform.resize(np.squeeze(self.model.predict(exp_input))[:,:,1], input.shape, mode='constant', preserve_range=True)

    def save(self, model_name):
        self.model.save_weights(model_name)

    def load(self, model_name):
        self.model.load_weights(model_name)

if __name__ == '__main__':

    model = SparseUnet(shape=(512, 512, 1))

    # model.train(train_dir='benchmark_EM/train',
    #             test_dir='benchmark_EM/test',
    #             out_dir='FINAL_512x512x4_200_epochs_only_sparse_run_3',
    #             epochs=200,
    #             batch_size=4,
    #             dense=False)

    # model.load('sparse_unet_epoch_118_val_loss_0.0000.hdf5')
    model.load('sparse_unet_epoch_193_val_loss_0.0000.hdf5')

    # img = tifffile.imread('../Data/CLEM_001/EM_001_50x50x50nm.tif')[80:90]

    img_lst = sorted(glob.glob(os.path.join('../Data/CLEM_003/EM04422_04_PNG', '*')))

    # img_pred = np.empty(img.shape)

    for i in tqdm(range(len(img_lst))):
        img_pred = model.predict(imageio.imread(img_lst[i]), tile_shape=(1167, 1167)) #(1167, 1167)

        imageio.imwrite(os.path.join('../Data/CLEM_003/EM0442_04_PNG_pred', 'pred_{}.png'.format(i)), img_pred.astype('float'))


    # tifffile.imwrite('EM_001_50x50x50nm_pred.tif', img_pred.astype('float32'))
    # img = imageio.imread('img.088.png')
    # img_pred = model.predict(img, tile_shape=(1167, 1167))
    # imageio.imwrite('img.088_pred.png', (img_pred > 0.8).astype('float'))

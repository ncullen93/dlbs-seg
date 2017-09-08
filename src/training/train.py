"""
Train a UNET model to predict a continuous 3D image from a given
3D continuous brain image.

The example here uses the input image as a target image (aka an 'Autoencoder') but the
target image can be any other brain image.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

try:
    this_file = os.path.dirname(os.path.realpath(__file__)).split(os.sep)
except:
    this_file = os.path.expanduser('~/desktop/projects/dlbs-seg/src/training/train.py').split(os.sep)

base_dir = os.sep.join(this_file[:-3])

data_dir = os.path.join(base_dir, 'data/')
results_dir = os.path.join(base_dir, 'results/')

# local imports
os.chdir(os.path.join(base_dir,'src/training/'))
from sampling import DataLoader, CSVDataset
from sampling import transforms as tx
from models import create_unet_model3D

from keras import callbacks as cbks


# tx.Compose lets you string together multiple transforms
co_tx = tx.Compose([tx.TypeCast('float32'),
                    tx.ExpandDims(axis=-1),
                    tx.RandomAffine(rotation_range=(-15,15), # rotate btwn -15 & 15 degrees
                                    translation_range=(0.1,0.1), # translate btwn -10% and 10% horiz, -10% and 10% vert
                                    shear_range=(-10,10), # shear btwn -10 and 10 degrees
                                    zoom_range=(0.85,1.15), # between 15% zoom-in and 15% zoom-out
                                    turn_off_frequency=5,
                                    fill_value='min',
                                    target_fill_mode='constant',
                                    target_fill_value=0) # how often to just turn off random affine transform (units=#samples)
                    ])

input_tx = tx.MinMaxScaler((-1,1)) # scale between -1 and 1

target_tx = tx.OneHot() # convert segmentation image to One-Hot representation for cross-entropy loss

# use a co-transform, meaning the same transform will be applied to input+target images at the same time 
# this is necessary since Affine transforms have random parameter draws which need to be shared
dataset = CSVDataset(filepath=os.path.join(data_dir,'t1seg_filemap_npy.csv'), 
                    base_path=os.path.join(data_dir,'preprocessed_npy'), # this path will be appended to all of the filenames in the csv file
                    input_cols=['T1'], # column in dataframe corresponding to inputs (can be an integer also)
                    target_cols=['T1-SEG'],# column in dataframe corresponding to targets (can be an integer also)
                    input_transform=input_tx, target_transform=target_tx, co_transform=co_tx,
                    co_transforms_first=True) # run co transforms before input/target transforms


# split into train and test set based on the `train-test` column in the csv file
# this splits alphabetically by values, and since 'test' comes before 'train' thus val_data is returned before train_data
train_data, val_data = dataset.train_test_split(test_size=0.3)

# overwrite co-transform on validation data so it doesnt have any random augmentation
val_data.set_co_transform(tx.Compose([tx.TypeCast('float32'),
                                      tx.ExpandDims(axis=-1)]))

# create a dataloader .. this is basically a keras DataGenerator -> can be fed to `fit_generator`
batch_size = 20
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# write an example batch to a folder as JPEG
#train_loader.write_a_batch(data_dir+'example_batch/')

n_labels = train_data[0][1].shape[-1]

# create model
model = create_unet_model3D(input_image_size=train_data[0][0].shape, n_labels=n_labels, layers=4,
                            lowest_resolution=8, mode='classification')

callbacks = [cbks.ModelCheckpoint(os.path.join(results_dir,'unet-model.h5'), 
                                  monitor='val_loss', save_best_only=True),
            cbks.ReduceLROnPlateau(monitor='val_loss', factor=0.1)]

model.fit_generator(generator=iter(train_loader), steps_per_epoch=np.ceil(len(train_data)/batch_size), 
                    epochs=200, verbose=1, callbacks=callbacks, 
                    shuffle=True, class_weight=None,
                    validation_data=iter(val_loader), validation_steps=np.ceil(len(val_data)/batch_size), 
                    max_queue_size=10, workers=1, use_multiprocessing=False,  initial_epoch=0)





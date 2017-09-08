"""
Preprocess raw DLBS data 
"""

import os
import shutil

import numpy as np
import pandas as pd

from tqdm import tqdm

import ants


def create_filemap(base_dir, save=True):
    """
    Args : 
        base_dir : string   
            full path to `dlbs-seg` directory
        save : boolean
            whether to save filemap

    Returns :
        pd.DataFrame
            the filemap

    Effects :
        saves filemap csv to base_dir/data/segmentation_filemap.csv
        note: this filemap can be used with raw or processed images,
        since it contains relative paths
    """
    data_dir = os.path.join(base_dir, 'data')
    raw_data_dir = os.path.join(data_dir, 'raw')

    # list all subjects in `images` directory
    raw_img_dir = os.path.join(raw_data_dir, 'images')
    img_subs = []
    for sub in os.listdir(raw_img_dir):
        if not sub.startswith('.'):
            img_subs.append(sub)
    img_subs = np.asarray(img_subs)
    
    # list all subjects in `segmentations` directory
    raw_seg_dir = os.path.join(raw_data_dir, 'segmentations')
    seg_subs = []
    for sub in os.listdir(raw_seg_dir):
        if (not sub.startswith('.')) and ('session' in sub):
            seg_subs.append(sub.split('_')[0])
    seg_subs = np.asarray(seg_subs)

    # subjects who have T1s and segmentations
    imgseg_subs = np.intersect1d(img_subs.astype('int'), 
                                 seg_subs.astype('int'))
    #print('Found %i subjects w/ t1 and segs' % len(imgseg_subs))

    # create t1 path - this will make sure they are ordered correctly
    t1paths = ['images/00%i/session_1/anat_1/anat.nii.gz'% iss for iss in imgseg_subs]
    segpaths = ['segmentations/%i_session1_T1_BrainSegmentation.nii.gz' % iss for iss in imgseg_subs]

    filemap = pd.DataFrame(np.vstack([t1paths,segpaths]).T, columns=['T1','T1-SEG'])
    filemap.index = imgseg_subs

    if save:
        filemap.to_csv(os.path.join(data_dir, 't1seg_filemap.csv'), index=True)

    # save npy version of filemap
    for i in range(filemap.shape[0]):
        filemap.iloc[i,0] = filemap.iloc[i,0].replace('.nii.gz', '.npy')
        filemap.iloc[i,1] = filemap.iloc[i,1].replace('.nii.gz', '.npy')

    if save:
        filemap.to_csv(os.path.join(data_dir, 't1seg_filemap_npy.csv'), index=True)


def _make_recursive_directories(base_dir, filepath):
    file_seps = filepath.split(os.sep)
    for i in range(len(file_seps)):
        new_dir = os.path.join(base_dir, '/'.join(file_seps[:i]))
        if not os.path.exists(new_dir):
            try:
                os.mkdir(new_dir)
            except:
                pass


def preprocess_images(base_dir):
    """
    Minimal preprocessing of t1 brain images before training the 
    deep learning model. 

    This function requires the `ANTsPy`
    package from `https://github.com/ANTsX/ANTsPy`

    This step consists of the following:
        - Intensity truncation
        - N4 Bias correction
        - resample to 2 mm^3 voxel resolution
        - save numpy array version of images
    
    Args : 
        base_dir : string   
            full path to `dlbs-seg` directory

    Returns :
        N/A

    Effects :
        save processed images into base_dir/data/preprocessed/

    """
    # get filemap of all the images
    filemap = pd.read_csv(os.path.join(base_dir, 'data/t1seg_filemap.csv'), index_col=0)

    base_load_path = os.path.join(base_dir, 'data/raw/')
    base_save_path = os.path.join(base_dir, 'data/preprocessed/')
    base_save_path_npy = os.path.join(base_dir,'data/preprocessed_npy/')

    if not os.path.exists(base_save_path):
        os.mkdir(base_save_path) 

    if not os.path.exists(base_save_path_npy):
        os.mkdir(base_save_path_npy) 

    t1_files = filemap['T1'].values
    seg_files = filemap['T1-SEG'].values
    for i in tqdm(range(len(t1_files))):
        t1_file = t1_files[i]
        seg_file = seg_files[i]
        # load image as float
        image = ants.image_read(os.path.join(base_load_path, t1_file), 
                                pixeltype='float')

        image_mask = ants.get_mask(image)

        # run N4 bias correction
        image = ants.n4_bias_field_correction(image, mask=image_mask)

        # downsample to 2mm^3 resolution
        image = image.resample_image((2,2,2), interp_type=0)

        # save t1 image to preprocessed directory
        _make_recursive_directories(base_save_path, t1_file)
        _make_recursive_directories(base_save_path_npy, t1_file)
        ants.image_write(image, os.path.join(base_save_path, t1_file))
        np.save(os.path.join(base_save_path_npy, t1_file.replace('.nii.gz','.npy')),
                image.numpy().astype('float32'))
        
        # load and save seg file
        seg_image = ants.image_read(os.path.join(base_load_path, seg_file))
        seg_image = seg_image.resample_image((2,2,2), interp_type=1)

        _make_recursive_directories(base_save_path, seg_file)
        _make_recursive_directories(base_save_path_npy, seg_file)
        ants.image_write(seg_image, os.path.join(base_save_path, seg_file))
        np.save(os.path.join(base_save_path_npy, seg_file.replace('.nii.gz','.npy')),
                seg_image.numpy().astype('uint8'))


if __name__=='__main__':
    project_dir = '/users/ncullen/Desktop/projects/dlbs-seg/'
    
    create_filemap(project_dir)
    preprocess_images(project_dir)

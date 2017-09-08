"""
Preprocess raw DLBS data 
"""

import os
import numpy as np
import pandas as pd

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
    proc_data_dir = os.path.join(data_dir, 'processed')

    if not os.path.exists(proc_data_dir):
        os.mkdir(proc_data_dir)

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
        if not sub.startswith('.'):
            seg_subs.append(sub)
    seg_subs = np.asarray(seg_subs)

    # subjects who have T1s and segmentations
    imgseg_subs = np.intersect1d(img_subs.astype('int'), 
                                 seg_subs.astype('int'))
    print('Found %i subjects w/ t1 and segs' % len(imgseg_subs))

    # create t1 path
    t1paths = ['images/00%i/session_1/anat_1/anat.nii.gz'% iss for iss in imgseg_subs]
    segpaths = ['segmentations/%i_session1_T1_BrainSegmentation.nii.gz' % iss for iss in imgseg_subs]

    filemap = pd.DataFrame(np.vstack([t1paths,segpaths]), columns=['T1','T1-SEG'])
    filemap.index = imgseg_subs

    filemap.to_csv(os.path.join(data_dir, 't1seg_filemap.csv'), index=True)
    return filemap


def _make_recursive_directories(base_dir, filepath):
    file_seps = filepath.split(os.sep)
    for i in range(len(file_seps)-1):
        new_dir = os.path.join(base_dir, file_seps)
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
        - Quick skull stripping
    
    Args : 
        base_dir : string   
            full path to `dlbs-seg` directory

    Returns :
        N/A

    Effects :
        save processed images into base_dir/data/preprocessed/

    """
    # get filemap of all the images
    filemap = create_filemap(base_dir, save=False)

    base_load_path = os.path.join(base_dir, 'data/raw/')
    base_save_path = os.path.join(base_dir, 'data/preprocessed/')
    if not os.path.exists(base_save_path):
        os.mkdir(base_save_path) 

    t1_files = filemap['T1']
    for t1_file in t1_files:
        # load image as float
        image = ants.image_read(os.path.join(base_load_path, t1_file), 
                                pixeltype='float')

        image_mask = ants.get_mask(image)

        # truncate intensity and run N4 bias correction
        image_n4 = ants.abp_n4(image, mask=image_mask)

        # mask image
        image_n4_skull = ants.mask_image(image_n4, image_n4_mask)

        # make sure appropriate directories exist in preprocessed dir
        _make_recursive_directories(base_save_path, t1_file)

        # save image to preprocessed directory
        ants.image_write(image_n4_skull, os.path.join(base_save_path, t1_file))
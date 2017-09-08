
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import fnmatch

import numpy as np
import pandas as pd
import PIL.Image as Image
import nibabel
from sklearn.model_selection import train_test_split

from . import transforms

try:
    import ants
    has_ants=True
except:
    has_ants=False

def nibabel_loader(path):
    return nibabel.load(path).get_data()
def ants_loader(path):
    return ants.image_read(path).numpy()

if has_ants:
    nifti_loader = ants_loader
else:
    nifti_loader = nibabel_loader

def default_file_reader(x, base_path=''):
    def pil_loader(path):
        return Image.open(path).convert('RGB')
    def npy_loader(path):
        return np.load(path)
    if isinstance(x, str):
        x = os.path.join(base_path, x)
        if x.endswith('.npy'):
            x = npy_loader(x)
        elif x.endswith('.nii.gz'):
            x = nifti_loader(x)
        else:
            try:
                x = pil_loader(x)
            except:
                raise ValueError('File Format is not supported')
    #else:
        #raise ValueError('x should be string, but got %s' % type(x))
    return x

def is_tuple_or_list(x):
    return isinstance(x, (tuple,list))

def _process_transform_argument(tform, num_inputs):
    tform = tform if tform is not None else PassThrough()
    if is_tuple_or_list(tform):
        if len(tform) != num_inputs:
            raise Exception('If transform is list, must provide one transform for each input')
        tform = [t if t is not None else PassThrough() for t in tform]
    else:
        tform = [tform] * num_inputs
    return tform

def _process_co_transform_argument(tform, num_inputs, num_targets):
    tform = tform if tform is not None else MultiArgPassThrough()
    if is_tuple_or_list(tform):
        if len(tform) != num_inputs:
            raise Exception('If transform is list, must provide one transform for each input')
        tform = [t if t is not None else MultiArgPassThrough() for t in tform]
    else:
        tform = [tform] * min(num_inputs, num_targets)
    return tform

def _process_csv_argument(csv):
    if isinstance(csv, str):
        df = pd.read_csv(csv)
    elif isinstance(csv, pd.DataFrame):
        df = csv
    else:
        raise ValueError('csv argument must be string or dataframe')
    return df

def _select_dataframe_columns(df, cols):
    if isinstance(cols[0], str):
        inputs = df.loc[:,cols].values
    elif isinstance(cols[0], int):
        inputs = df.iloc[:,cols].values
    else:
        raise ValueError('Provided columns should be string column names or integer column indices')
    return inputs

def _process_cols_argument(cols):
    if isinstance(cols, tuple):
        cols = list(cols)
    return cols

def _return_first_element_of_list(x):
    return x[0]

class PassThrough(object):
    def transform(self, X):
        return X

class MultiArgPassThrough(object):
    def transform(self, *x):
        return x


def _process_array_argument(x):
    if not is_tuple_or_list(x):
        x = [x]
    return x


class BaseDataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __len__(self):
        return len(self.inputs) if not isinstance(self.inputs, (tuple,list)) else len(self.inputs[0])

    def add_input_transform(self, transform, add_to_front=True, idx=None):
        if idx is None:
            idx = np.arange(len(self.num_inputs))
        elif not is_tuple_or_list(idx):
            idx = [idx]

        if add_to_front:
            for i in idx:
                self.input_transform[i] = transforms.Compose([transform, self.input_transform[i]])
        else:
            for i in idx:
                self.input_transform[i] = transforms.Compose([self.input_transform[i], transform])

    def add_target_transform(self, transform, add_to_front=True, idx=None):
        if idx is None:
            idx = np.arange(len(self.num_targets))
        elif not is_tuple_or_list(idx):
            idx = [idx]

        if add_to_front:
            for i in idx:
                self.target_transform[i] = transforms.Compose([transform, self.target_transform[i]])
        else:
            for i in idx:
                self.target_transform[i] = transforms.Compose([self.target_transform[i], transform])

    def add_co_transform(self, transform, add_to_front=True, idx=None):
        if idx is None:
            idx = np.arange(len(self.min_inputs_or_targets))
        elif not is_tuple_or_list(idx):
            idx = [idx]

        if add_to_front:
            for i in idx:
                self.co_transform[i] = transforms.Compose([transform, self.co_transform[i]])
        else:
            for i in idx:
                self.co_transform[i] = transforms.Compose([self.co_transform[i], transform])

    def load(self, num_samples=None, load_range=None, verbose=0):
        """
        Load all data or a subset of the data into actual memory.
        For instance, if the inputs are paths to image files, then this
        function will actually load those images.
    
        Arguments
        ---------
        num_samples : integer (optional)
            number of samples to load. if None, will load all
        load_range : numpy array of integers (optional)
            the index range of images to load
            e.g. np.arange(4) loads the first 4 inputs+targets
        """
        def _parse_shape(x):
            if isinstance(x, (list,tuple)):
                return (len(x),)
            elif isinstance(x, np.ndarray):
                return x.shape
            else:
                return (1,)

        if num_samples is None and load_range is None:
            num_samples = len(self)
            load_range = np.arange(num_samples)
        elif num_samples is None and load_range is not None:
            num_samples = len(load_range)
        elif num_samples is not None and load_range is None:
            load_range = np.arange(num_samples)


        if self.has_target:
            for enum_idx, sample_idx in enumerate(load_range):
                if verbose > 0:
                    if (enum_idx+1) % int(len(load_range)/5) == 0:
                        print('Loading sample %i / %i' %(enum_idx, len(load_range)))
                input_sample, target_sample = self.__getitem__(sample_idx)

                if enum_idx == 0:
                    if self.num_inputs == 1:
                        inputs = np.empty((len(load_range), *_parse_shape(input_sample))).astype('float32')
                    else:
                        inputs = [np.empty((len(load_range), *_parse_shape(input_sample[i]))).astype('float32') for i in range(self.num_inputs)]

                    if self.num_targets == 1:
                        targets = np.empty((len(load_range), *_parse_shape(target_sample))).astype('float32')
                    else:
                        targets = [np.empty((len(load_range), *_parse_shape(target_sample[i]))).astype('float32') for i in range(self.num_targets)]

                if self.num_inputs == 1:
                    inputs[enum_idx] = input_sample
                else:
                    for i in range(self.num_inputs):
                        inputs[i][enum_idx] = input_sample[i]

                if self.num_targets == 1:
                    targets[enum_idx] = target_sample
                else:
                    for i in range(self.num_targets):
                        targets[i][enum_idx] = target_sample[i]

            return inputs, targets
        else:
            for enum_idx, sample_idx in enumerate(load_range):
                if verbose > 0:
                    if (enum_idx+1) % int(len(load_range)/5) == 0:
                        print('Loading sample %i / %i' %(enum_idx, len(load_range)))
                input_sample = self.__getitem__(sample_idx)

                if enum_idx == 0:
                    if self.num_inputs == 1:
                        inputs = np.empty((len(load_range), *_parse_shape(input_sample))).astype('float32')
                    else:
                        inputs = [np.empty((len(load_range), *_parse_shape(input_sample[i]))).astype('float32') for i in range(self.num_inputs)]

                if self.num_inputs == 1:
                    inputs[enum_idx] = input_sample
                else:
                    for i in range(self.num_inputs):
                        inputs[i][enum_idx] = input_sample[i]

            return inputs

    def fit_transforms(self):
        """
        Make a single pass through the entire dataset in order to fit 
        any parameters of the transforms which require the entire dataset.
        e.g. StandardScaler() requires mean and std for the entire dataset.

        If you dont call this fit function, then transforms which require properties
        of the entire dataset will just work at the batch level.
        e.g. StandardScaler() will normalize each batch by the specific batch mean/std
        """
        it_fit = hasattr(self.input_transform, 'update_fit')
        tt_fit = hasattr(self.target_transform, 'update_fit')
        ct_fit = hasattr(self.co_transform, 'update_fit')
        if it_fit or tt_fit or ct_fit:
            for sample_idx in range(len(self)):
                if hasattr(self, 'input_loader'):
                    x = self.input_loader(self.inputs[sample_idx])
                else:
                    x = self.inputs[sample_idx]
                if it_fit:
                    self.input_transform.update_fit(x)
                if self.has_target:
                    if hasattr(self, 'target_loader'):
                        y = self.target_loader(self.targets[sample_idx])
                    else:
                        y = self.targets[sample_idx]
                if tt_fit:
                    self.target_transform.update_fit(y)
                if ct_fit:
                    self.co_transform.update_fit(x,y)


class ArrayDataset(BaseDataset):

    def __init__(self,
                 inputs,
                 targets=None,
                 input_transform=None, 
                 target_transform=None,
                 co_transform=None):
        """
        Dataset class for loading in-memory data.

        Arguments
        ---------
        inputs: numpy array

        targets : numpy array

        input_transform : class with __call__ function implemented
            transform to apply to input sample individually

        target_transform : class with __call__ function implemented
            transform to apply to target sample individually

        co_transform : class with __call__ function implemented
            transform to apply to both input and target sample simultaneously

        """
        self.inputs = inputs if is_tuple_or_list(inputs) else [inputs]
        self.num_inputs = len(self.inputs)
        self.input_return_processor = _return_first_element_of_list if self.num_inputs==1 else PassThrough()

        if targets is None:
            self.has_target = False
        else:
            self.targets = targets if is_tuple_or_list(targets) else [targets]
            self.num_targets = len(self.targets)
            self.target_return_processor = _return_first_element_of_list if self.num_targets==1 else PassThrough()
            self.min_inputs_or_targets = min(self.num_inputs, self.num_targets)
            self.has_target = True            
        
        self.input_transform = _process_transform_argument(input_transform, self.num_inputs)
        if self.has_target:
            self.target_transform = _process_transform_argument(target_transform, self.num_targets)
            self.co_transform = _process_co_transform_argument(co_transform, self.num_inputs, self.num_targets)

    def __getitem__(self, index):
        """
        Index the dataset and return the input + target
        """
        input_sample = [self.input_transform[i].transform(self.inputs[i][index]) for i in range(self.num_inputs)]

        if self.has_target:
            target_sample = [self.target_transform[i].transform(self.targets[i][index]) for i in range(self.num_targets)]
            for i in range(self.min_inputs_or_targets):
                input_sample[i], target_sample[i] = self.co_transform[i].transform(input_sample[i], target_sample[i])

            return self.input_return_processor(input_sample), self.target_return_processor(target_sample)
        else:
            return self.input_return_processor(input_sample)


class CSVDataset(BaseDataset):

    def __init__(self,
                 filepath,
                 input_cols=[0],
                 target_cols=[1],
                 input_transform=None,
                 target_transform=None,
                 co_transform=None,
                 co_transforms_first=False,
                 base_path=None):
        """
        Initialize a Dataset from a CSV file/dataframe. This does NOT
        actually load the data into memory if the CSV contains filepaths.

        Arguments
        ---------
        csv : string or pandas.DataFrame
            if string, should be a path to a .csv file which
            can be loaded as a pandas dataframe
        
        input_cols : int/list of ints, or string/list of strings
            which columns to use as input arrays.
            If int(s), should be column indicies
            If str(s), should be column names 
        
        target_cols : int/list of ints, or string/list of strings
            which columns to use as input arrays.
            If int(s), should be column indicies
            If str(s), should be column names 

        input_transform : class which implements a __call__ method
            tranform(s) to apply to inputs during runtime loading

        target_tranform : class which implements a __call__ method
            transform(s) to apply to targets during runtime loading

        co_transform : class which implements a __call__ method
            transform(s) to apply to both inputs and targets simultaneously
            during runtime loading
        """
        self.input_cols = _process_cols_argument(input_cols)
        self.target_cols = _process_cols_argument(target_cols)
        self.base_path = '' if base_path is None else base_path
        
        self.df = _process_csv_argument(filepath)

        self.inputs = _select_dataframe_columns(self.df, input_cols)
        self.num_inputs = self.inputs.shape[1]
        self.input_return_processor = _return_first_element_of_list if self.num_inputs==1 else PassThrough()
        self.co_transforms_first = co_transforms_first

        if target_cols is None:
            self.num_targets = 0
            self.has_target = False
        else:
            self.targets = _select_dataframe_columns(self.df, target_cols)
            self.num_targets = self.targets.shape[1]
            self.target_return_processor = _return_first_element_of_list if self.num_targets==1 else PassThrough()
            self.has_target = True

            self.min_inputs_or_targets = min(self.num_inputs, self.num_targets)

        self.input_loader = default_file_reader
        self.target_loader = default_file_reader
        
        self.input_transform = _process_transform_argument(input_transform, self.num_inputs)
        if self.has_target:
            self.target_transform = _process_transform_argument(target_transform, self.num_targets)
            self.co_transform = _process_co_transform_argument(co_transform, self.num_inputs, self.num_targets)

    def set_input_transform(self, transform):
        """ overwrite input transform correctly """
        self.input_transform = _process_transform_argument(transform, self.num_inputs)
    def set_target_transform(self, transform):
        """ overwrite target transform correctly """
        self.target_transform = _process_transform_argument(transform, self.num_targets)
    def set_co_transform(self, transform):
        """ overwrite co transform correctly """
        if not self.has_target:
            raise ValueError('dataset must have a target tensor')
        self.co_transform = _process_co_transform_argument(transform, self.num_inputs, self.num_targets)

    def __getitem__(self, index):
        """
        Index the dataset and return the input + target
        """
        if self.co_transforms_first:
            input_sample = [self.input_loader(self.inputs[index, i], self.base_path) for i in range(self.num_inputs)]

            if self.has_target:
                target_sample = [self.target_loader(self.targets[index, i], self.base_path) for i in range(self.num_targets)]
                for i in range(self.min_inputs_or_targets):
                    input_sample[i], target_sample[i] = self.co_transform[i].transform(input_sample[i], target_sample[i])

                for i in range(len(input_sample)):
                    input_sample[i] = self.input_transform[i].transform(input_sample[i])
                    target_sample[i] = self.target_transform[i].transform(target_sample[i])

                return self.input_return_processor(input_sample), self.target_return_processor(target_sample)
            else:
                for i in range(len(input_sample)):
                    input_sample[i] = self.input_transform[i].transform(input_sample[i])
                return self.input_return_processor(input_sample)  
        else:
            input_sample = [self.input_transform[i].transform(self.input_loader(self.inputs[index, i], self.base_path)) for i in range(self.num_inputs)]

            if self.has_target:
                target_sample = [self.target_transform[i].transform(self.target_loader(self.targets[index, i], self.base_path)) for i in range(self.num_targets)]
                for i in range(self.min_inputs_or_targets):
                    input_sample[i], target_sample[i] = self.co_transform[i].transform(input_sample[i], target_sample[i])

                return self.input_return_processor(input_sample), self.target_return_processor(target_sample)
            else:
                return self.input_return_processor(input_sample) 

    def split_by_column(self, col):
        """
        Split this dataset object into multiple dataset objects based on 
        the unique factors of the given column. The number of returned
        datasets will be equal to the number of unique values in the given
        column. The transforms and original dataframe will all be transferred
        to the new datasets 

        Useful for splitting a dataset into train/val/test datasets.

        Arguments
        ---------
        col : integer or string
            which column to split the data on. 
            if int, should be column index
            if str, should be column name

        Returns
        -------
        - list of new datasets with transforms copied
        """
        if isinstance(col, int):
            split_vals = self.df.iloc[:,col].values.flatten()

            new_df_list = []
            for unique_split_val in np.unique(split_vals):
                new_df = self.df[:][self.df.iloc[:,col]==unique_split_val]
                new_df_list.append(new_df)
        elif isinstance(col, str):
            split_vals = self.df.loc[:,col].values.flatten()

            new_df_list = []
            for unique_split_val in np.unique(split_vals):
                new_df = self.df[:][self.df.loc[:,col]==unique_split_val]
                new_df_list.append(new_df)
        else:
            raise ValueError('col argument not valid - must be column name or index')

        new_datasets = []
        for new_df in new_df_list:
            new_dataset = self.copy(new_df)
            new_datasets.append(new_dataset)

        return new_datasets

    def train_test_split(self, test_size):
        df = self.df
        train_df, test_df = train_test_split(df, test_size=test_size)

        return self.copy(df=train_df), self.copy(df=test_df)

    def copy(self, df=None):
        if df is None:
            df = self.df

        return CSVDataset(df,
                          input_cols=self.input_cols, 
                          target_cols=self.target_cols,
                          input_transform=self.input_transform,
                          target_transform=self.target_transform,
                          co_transform=self.co_transform,
                          co_transforms_first=self.co_transforms_first,
                          base_path=self.base_path)


class FolderDataset(BaseDataset):

    def __init__(self, 
                 root,
                 class_mode='label',
                 input_regex='*',
                 target_regex=None,
                 input_transform=None, 
                 target_transform=None,
                 co_transform=None, 
                 input_loader='npy'):
        """
        Dataset class for loading out-of-memory data.

        Arguments
        ---------
        root : string
            path to main directory

        class_mode : string in `{'label', 'image'}`
            type of target sample to look for and return
            `label` = return class folder as target
            `image` = return another image as target as found by 'target_regex'
                NOTE: if class_mode == 'image', you must give an
                input and target regex and the input/target images should
                be in a folder together with no other images in that folder

        input_regex : string (default is any valid image file)
            regular expression to find input images
            e.g. if all your inputs have the word 'input', 
            you'd enter something like input_regex='*input*'
        
        target_regex : string (default is Nothing)
            regular expression to find target images if class_mode == 'image'
            e.g. if all your targets have the word 'segment', 
            you'd enter somthing like target_regex='*segment*'

        transform : transform class
            transform to apply to input sample individually

        target_transform : transform class
            transform to apply to target sample individually

        input_loader : string in `{'npy', 'pil', 'nifti'} or callable
            defines how to load samples from file
            if a function is provided, it should take in a file path
            as input and return the loaded sample.

        """
        self.input_loader = default_file_reader
        self.target_loader = default_file_reader if class_mode == 'image' else lambda x: x

        root = os.path.expanduser(root)

        classes, class_to_idx = _find_classes(root)
        inputs, targets = _finds_inputs_and_targets(root, class_mode,
            class_to_idx, input_regex, target_regex)

        if len(inputs) == 0:
            raise(RuntimeError('Found 0 images in subfolders of: %s' % root))
        else:
            print('Found %i images' % len(inputs))

        self.root = os.path.expanduser(root)
        self.inputs = inputs
        self.targets = targets
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.input_transform = input_transform if input_transform is not None else lambda x: x
        if isinstance(input_transform, (tuple,list)):
            self.input_transform = transforms.Compose(self.input_transform)
        self.target_transform = target_transform if target_transform is not None else lambda x: x
        if isinstance(target_transform, (tuple,list)):
            self.target_transform = transforms.Compose(self.target_transform)
        self.co_transform = co_transform if co_transform is not None else lambda x,y: (x,y)
        if isinstance(co_transform, (tuple,list)):
            self.co_transform = transforms.Compose(self.co_transform)
        
        self.class_mode = class_mode

    def get_full_paths(self):
        return [os.path.join(self.root, i) for i in self.inputs]

    def __getitem__(self, index):
        input_sample = self.inputs[index]
        input_sample = self.input_loader(input_sample)
        input_sample = self.input_transform(input_sample)

        target_sample = self.targets[index]
        target_sample = self.target_loader(target_sample)
        target_sample = self.target_transform(target_sample)
        
        input_sample, target_sample = self.co_transform(input_sample, target_sample)

        return input_sample, target_sample
    
    def __len__(self):
        return len(self.inputs)



def _find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def _is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        '.nii.gz', '.npy'
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def _finds_inputs_and_targets(directory, class_mode, class_to_idx=None, 
            input_regex=None, target_regex=None, ):
    """
    Map a dataset from a root folder
    """
    if class_mode == 'image':
        if not input_regex and not target_regex:
            raise ValueError('must give input_regex and target_regex if'+
                ' class_mode==image')
    inputs = []
    targets = []
    for subdir in sorted(os.listdir(directory)):
        d = os.path.join(directory, subdir)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if _is_image_file(fname):
                    if fnmatch.fnmatch(fname, input_regex):
                        path = os.path.join(root, fname)
                        inputs.append(path)
                        if class_mode == 'label':
                            targets.append(class_to_idx[subdir])
                    if class_mode == 'image' and \
                            fnmatch.fnmatch(fname, target_regex):
                        path = os.path.join(root, fname)
                        targets.append(path)
    if class_mode is None:
        return inputs
    else:
        return inputs, targets

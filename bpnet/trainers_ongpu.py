from kipoi_utils.external.flatten_json import flatten
from bpnet.utils import write_json, dict_prefix_key
from bpnet.data import NumpyDataset
from keras.callbacks import EarlyStopping, CSVLogger, TensorBoard
from collections import OrderedDict
import os
import gin
import pandas as pd
from tqdm import tqdm
import logging
import tensorflow as tf
import numpy as np
import inspect
from kipoi_utils.external.torch.sampler import BatchSampler
import collections
from kipoi_utils.data_utils import (numpy_collate, numpy_collate_concat, get_dataset_item,
                                    DataloaderIterable, batch_gen, get_dataset_lens, iterable_cycle)
from copy import deepcopy
from bpnet.utils import flatten, unflatten
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
from kipoi.writers import HDF5BatchWriter
from kipoi.readers import HDF5Reader

''' imports needed for to_numpy, normalize, normalize_column '''
try:
    import torch
    from torch.utils.data import DataLoader
    USE_TORCH = True
except Exception:
    # use the Kipoi dataloader as a fall-back strategy
    from kipoi.data import DataLoader
    USE_TORCH = False
import abc

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class BatchNormDataset(NumpyDataset):

    """Data-structure of arbitrarily nested arrays
       with the same first axis
    """
    
    # Function to normalize specific column in a minibatch
    def normalize_column(self, np_row, col):
        row = np_row[:,:,col]
        mean = np.mean(row, axis=0)
        var = np.var(row, axis=0)
        row = np.subtract(row, mean)
        return np.divide(row, np.sqrt(var) + 1e-6)

    # Function to normalize minibatch
    def normalize(self, data):
        # Turn data to numpy array
        np_row = np.array(data['seq'])

        # Normalize each specified column
        for i in range(self.batchnorm_begin, self.batchnorm_end+1):
            if(self.print_test is True):
                print("We are batchnorming column ", i)
                print("Before batchnorm: ", data['seq'][:,:,i])
            data['seq'][:,:,i] = self.normalize_column(np_row, i)
            if(self.print_test is True):
                print("After batchnorm: ", data['seq'][:,:,i])

        return data

    def to_numpy(self, data):
        if not USE_TORCH:
            return data
        if isinstance(data, torch.Tensor):
            return data.numpy()
        elif isinstance(data, collections.Mapping):
            return {key: self.to_numpy(data[key]) for key in data}
        elif isinstance(data, collections.Sequence):
            if isinstance(data[0], str):
                return data
            else:
                return [self.to_numpy(sample) for sample in data]
        else:
            raise ValueError("Leafs of the nested structure need to be numpy arrays")

    ''' Overriding functions of Dataset '''
    def batch_train_iter(self, cycle=True, **kwargs):
        """Returns samples directly useful for training the model:
        (x["inputs"],x["targets"])
        Args:
          cycle: when True, the returned iterator will run indefinitely go through the dataset
            Use True with `fit_generator` in Keras.
          **kwargs: Arguments passed to self.batch_iter(**kwargs)
        """
        
        if cycle:
            return ((self.normalize(self.to_numpy(x["inputs"])), self.to_numpy(x["targets"]))
                    for x in iterable_cycle(self._batch_iterable(**kwargs)))
        else:
            return ((self.normalize(x["inputs"]), x["targets"]) for x in self.batch_iter(**kwargs))

    def __init__(self, np_dataset, batchnorm_begin, batchnorm_end, attrs=None, print_test=False):
        """
        Args:
          data: any arbitrarily nested dict/list of np.arrays
            with the same first axis size
          attrs: optional dictionary of attributes
        """
        self.data = np_dataset.data
        self.batchnorm_begin = batchnorm_begin
        self.batchnorm_end = batchnorm_end
        self.print_test = print_test

        if attrs is None:
            self.attrs = OrderedDict()
        else:
            self.attrs = attrs

        self._validate()

    def _validate(self):
        # Make sure the first axis is the same
        # for k,v in flatten(data).items():
        assert len(set(self.get_lens())) == 1

    def get_lens(self):
        return list(flatten(self.dapply(len)).values())

    def __len__(self):
        return self.get_lens()[0]

    def __getitem__(self, idx):
        def get_item(arr, idx):
            return arr[idx]
        return self.dapply(get_item, idx=idx)

    def loc(self, idx):
        return super().__init__(self[idx], attrs=deepcopy(self.attrs))

    def flatten(self):
        return super().__init__(flatten(self.data), attrs=deepcopy(self.attrs))

    def unflatten(self):
        return super().__init__(unflatten(self.data), attrs=deepcopy(self.attrs))

    def shapes(self):
        from pprint import pprint

        def get_shape(arr):
            return str(arr.shape)

        out = self.dapply(get_shape)
        pprint(out)

    def dapply(self, fn, *args, **kwargs):
        """Apply a function to each element in the list
        Returns a nested dictionary
        """
        def _dapply(data, fn, *args, **kwargs):
            if type(data).__module__ == 'numpy':
                return fn(data, *args, **kwargs)
            elif isinstance(data, collections.Mapping):
                return {key: _dapply(data[key], fn, *args, **kwargs) for key in data}
            elif isinstance(data, collections.Sequence):
                return [_dapply(sample, fn, *args, **kwargs) for sample in data]
            else:
                raise ValueError("Leafs of the nested structure need to be numpy arrays")

        return _dapply(self.data, fn, *args, **kwargs)

    def sapply(self, fn, *args, **kwargs):
        """Same as dapply but returns NumpyDataset
        """
        return super().__init__(self.dapply(fn, *args, **kwargs), deepcopy(self.attrs))

    def aggregate(self, fn=np.mean, axis=0):
        """Aggregate across all tracks
        Args:
          idx: subset index
        """
        return self.dapply(fn, axis=axis)

    def shuffle(self):
        """Permute the order of seqlets
        """
        idx = pd.Series(np.arange(len(self))).sample(frac=1).values
        return self.loc(idx)

    def split(self, i):
        """Split the Dataset at a certain index
        """
        return self.loc(np.arange(i)), self.loc(np.arange(i, len(self)))

    def append(self, datax):
        """Append two datasets
        """
        return super().__init__(data=numpy_collate_concat([self.data, datax.data]),
                                attrs=deepcopy(self.attrs))

    def save(self, file_path, **kwargs):
        """Save the dataset to an hdf5 file
        """
        obj = HDF5BatchWriter(file_path=file_path, **kwargs)
        obj.batch_write(self.data)
        # Store the attrs
        for k, v in self.attrs.items():
            obj.f.attrs[k] = v
        obj.close()

    @classmethod
    def load(cls, file_path):
        """Load the dataset from an hdf5 dataset
        """
        with HDF5Reader(file_path) as obj:
            data = obj.load_all()
            attrs = OrderedDict(obj.f.attrs)
        return cls(data, attrs)

    @classmethod
    def concat(cls, objects):
        return cls(data=numpy_collate_concat(objects), attrs=None)

@gin.configurable
class GpuSeqModelTrainer:
    def __init__(self, model, train_dataset, valid_dataset, output_dir,
                 cometml_experiment=None, wandb_run=None, batchnorm_begin=-1, batchnorm_end=-1, print_test=False):
        """
        Args:
          model: compiled keras.Model
          train: training Dataset (object inheriting from kipoi.data.Dataset)
          valid: validation Dataset (object inheriting from kipoi.data.Dataset)
          output_dir: output directory where to log the training
          cometml_experiment: if not None, append logs to commetml
        """
        
        """
        My added Args:
          batchnorm_begin: index to begin batch norming (0th index)
          batchnorm_end: index to stop batch norming (0th index, inclusive)
          print_test: print information about batchnorming and the resulting datasets
        """
        
        # override the model class
        self.seq_model = model
        self.model = self.seq_model.model

        # Convert the given datasets (of type NumpyDataset) to our own class, which batchnorms input
        self.train_dataset = BatchNormDataset(train_dataset, batchnorm_begin, batchnorm_end, print_test)
        self.valid_dataset = [(valid_dataset[0][0], OurNumpyDataset(valid_dataset[0][1], batchnorm_begin, batchnorm_end)), (valid_dataset[1][0], OurNumpyDataset(valid_dataset[1][1], batchnorm_begin, batchnorm_end))]

        # Sanity Check
        if(print_test is True):
            print("First item of train dataset: ", train_dataset[1])
            print("First item of valid dataset: ", valid_dataset[1][1][1])

        self.cometml_experiment = cometml_experiment
        self.wandb_run = wandb_run
        self.metrics = dict()

        if not isinstance(self.valid_dataset, list):
            # package the validation dataset into a list of validation datasets
            self.valid_dataset = [('valid', self.valid_dataset)]

        # setup the output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ckp_file = f"{self.output_dir}/model.h5"
        if os.path.exists(self.ckp_file):
            raise ValueError(f"model.h5 already exists in {self.output_dir}")
        self.history_path = f"{self.output_dir}/history.csv"
        self.evaluation_path = f"{self.output_dir}/evaluation.valid.json"

    def train(self,
              batch_size=256,
              epochs=100,
              early_stop_patience=4,
              num_workers=8,
              train_epoch_frac=1.0,
              valid_epoch_frac=1.0,
              train_samples_per_epoch=None,
              validation_samples=None,
              train_batch_sampler=None,
              tensorboard=True):
        """Train the model
        Args:
          batch_size:
          epochs:
          patience: early stopping patience
          num_workers: how many workers to use in parallel
          train_epoch_frac: if smaller than 1, then make the epoch shorter
          valid_epoch_frac: same as train_epoch_frac for the validation dataset
          train_batch_sampler: batch Sampler for training. Useful for say Stratified sampling
          tensorboard: if True, tensorboard output will be added
        """

        if train_batch_sampler is not None:
            train_it = self.train_dataset.batch_train_iter(shuffle=False,
                                                           batch_size=1,
                                                           drop_last=None,
                                                           batch_sampler=train_batch_sampler,
                                                           num_workers=num_workers)
        else:
            train_it = self.train_dataset.batch_train_iter(batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers)
        count = 0

        next(train_it)
        valid_dataset = self.valid_dataset[0][1]  # take the first one
        valid_it = valid_dataset.batch_train_iter(batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers)
        next(valid_it)

        if tensorboard:
            tb = [TensorBoard(log_dir=self.output_dir)]
        else:
            tb = []

        if self.wandb_run is not None:
            from wandb.keras import WandbCallback
            wcp = [WandbCallback(save_model=False)]  # we save the model using ModelCheckpoint
        else:
            wcp = []

        # train the model
        if len(valid_dataset) == 0:
            raise ValueError("len(self.valid_dataset) == 0")

        if train_samples_per_epoch is None:
            train_steps_per_epoch = max(int(len(self.train_dataset) / batch_size * train_epoch_frac), 1)
        else:
            train_steps_per_epoch = max(int(train_samples_per_epoch / batch_size), 1)

        if validation_samples is None:
            # parametrize with valid_epoch_frac
            validation_steps = max(int(len(valid_dataset) / batch_size * valid_epoch_frac), 1)
        else:
            validation_steps = max(int(validation_samples / batch_size), 1)

        # where the error is
        #print(self.train_dataset)
        self.model.fit_generator(
            train_it,
            epochs=epochs,
            steps_per_epoch=train_steps_per_epoch,
            validation_data=valid_it,
            validation_steps=validation_steps,
            callbacks=[
                EarlyStopping(
                    patience=early_stop_patience,
                    restore_best_weights=True
                ),
                CSVLogger(self.history_path)
            ] + tb + wcp
        )
        self.model.save(self.ckp_file)

        # log metrics from the best epoch
        try:
            dfh = pd.read_csv(self.history_path)
            m = dict(dfh.iloc[dfh.val_loss.idxmin()])
            if self.cometml_experiment is not None:
                self.cometml_experiment.log_metrics(m, prefix="best-epoch/")
            if self.wandb_run is not None:
                self.wandb_run.summary.update(flatten(dict_prefix_key(m, prefix="best-epoch/"), separator='/'))
        except FileNotFoundError as e:
            logger.warning(e)

    def evaluate(self, metric, batch_size=256, num_workers=8, eval_train=False, eval_skip=[], save=True, **kwargs):
        """Evaluate the model on the validation set
        Args:
          metric: a function accepting (y_true, y_pred) and returning the evaluation metric(s)
          batch_size:
          num_workers:
          eval_train: if True, also compute the evaluation metrics on the training set
          save: save the json file to the output directory
        """
        if len(kwargs) > 0:
            logger.warn(f"Extra kwargs were provided to trainer.evaluate(): {kwargs}")
        # Save the complete model -> HACK
        self.seq_model.save(os.path.join(self.output_dir, 'seq_model.pkl'))

        # contruct a list of dataset to evaluate
        if eval_train:
            eval_datasets = [('train', self.train_dataset)] + self.valid_dataset
        else:
            eval_datasets = self.valid_dataset

        # skip some datasets for evaluation
        try:
            if len(eval_skip) > 0:
                logger.info(f"Using eval_skip: {eval_skip}")
                eval_datasets = [(k, v) for k, v in eval_datasets if k not in eval_skip]
        except Exception:
            logger.warn(f"eval datasets don't contain tuples. Unable to skip them using {eval_skip}")

        metric_res = OrderedDict()
        for d in eval_datasets:
            if len(d) == 2:
                dataset_name, dataset = d
                eval_metric = None  # Ignore the provided metric
            elif len(d) == 3:
                # specialized evaluation metric was passed
                dataset_name, dataset, eval_metric = d
            else:
                raise ValueError("Valid dataset needs to be a list of tuples of 2 or 3 elements"
                                 "(name, dataset) or (name, dataset, metric)")
            logger.info(f"Evaluating dataset: {dataset_name}")
            metric_res[dataset_name] = self.seq_model.evaluate(dataset,
                                                               eval_metric=eval_metric,
                                                               num_workers=num_workers,
                                                               batch_size=batch_size)
        if save:
            write_json(metric_res, self.evaluation_path, indent=2)
            logger.info("Saved metrics to {}".format(self.evaluation_path))

        if self.cometml_experiment is not None:
            self.cometml_experiment.log_metrics(flatten(metric_res, separator='/'), prefix="eval/")

        if self.wandb_run is not None:
            self.wandb_run.summary.update(flatten(dict_prefix_key(metric_res, prefix="eval/"), separator='/'))

        return metric_res

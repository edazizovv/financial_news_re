#

import numpy
import torch


class Samples:
    @staticmethod
    def sample(any_categorical, any_numerical, categorical, numerical, output, categorical_embeddings, numerical_shape, n_classes, test_partition=0.2, validation_partition=0.2, verbose=False):

        # Pytorch Data Preparation

        if any_categorical:
            categorical = torch.tensor(categorical, dtype=torch.int64)
        else:
            categorical = None
        if any_numerical:
            numerical = torch.tensor(numerical, dtype=torch.float)
        else:
            numerical = None
        output = torch.tensor(output).flatten()

        if verbose:
            if any_categorical:
                print(categorical.shape)
            else:
                print(0)
            if any_numerical:
                print(numerical.shape)
            else:
                print(0)
            print(output.shape)

        # Train-Test Split

        samples = [int(test_partition * len(output))]
        samples = [len(output) - samples[0]] + samples
        a = numpy.array(numpy.arange(len(output)))
        a_train = numpy.random.choice(a, size=samples[0], replace=False)
        a_test = numpy.setdiff1d(a, a_train)

        if any_categorical:
            categorical_data_train, categorical_data_test = categorical[a_train], categorical[a_test]
        else:
            categorical_data_train, categorical_data_test = None, None
        if any_numerical:
            numerical_data_train, numerical_data_test = numerical[a_train], numerical[a_test]
        else:
            numerical_data_train, numerical_data_test = None, None
        output_data_train, output_data_test = output[a_train], output[a_test]

        if verbose:
            if any_categorical:
                print(categorical_data_train.shape, categorical_data_test.shape)
            else:
                print(0, 0)
            if any_numerical:
                print(numerical_data_train.shape, numerical_data_test.shape)
            else:
                print(0, 0)
            print(output_data_train.shape, output_data_test.shape)

        # Train-Validation Split

        samples = [int(validation_partition * len(output_data_train))]
        samples = [len(output_data_train) - samples[0]] + samples
        a = numpy.array(numpy.arange(len(output_data_train)))
        a_train = numpy.random.choice(a, size=samples[0], replace=False)
        a_validation = numpy.setdiff1d(a, a_train)

        if any_categorical:
            categorical_data_train, categorical_data_validation = categorical_data_train[a_train], categorical_data_train[a_validation]
        else:
            categorical_data_train, categorical_data_validation = None, None
        if any_numerical:
            numerical_data_train, numerical_data_validation = numerical_data_train[a_train], numerical_data_train[a_validation]
        else:
            numerical_data_train, numerical_data_validation = None, None
        output_data_train, output_data_validation = output_data_train[a_train], output_data_train[a_validation]

        if verbose:
            if any_categorical:
                print(categorical_data_train.shape, categorical_data_validation.shape, categorical_data_test.shape)
            else:
                print(0, 0)
            if any_numerical:
                print(numerical_data_train.shape, numerical_data_validation.shape, numerical_data_test.shape)
            else:
                print(0, 0)
            print(output_data_train.shape, output_data_validation.shape, output_data_test.shape)

        train = DataFormats(any_categorical, any_numerical, categorical_data_train, numerical_data_train, output_data_train)
        validation = DataFormats(any_categorical, any_numerical, categorical_data_validation, numerical_data_validation, output_data_validation)
        test = DataFormats(any_categorical, any_numerical, categorical_data_test, numerical_data_test, output_data_test)

        return train, validation, test, categorical_embeddings, numerical_shape, n_classes


class DataFormats:
    def __init__(self, any_categorical, any_numerical, categorical=None, numerical=None, output=None, categorical_embeddings=[], n_classes=None):
        self.any_categorical = any_categorical
        self.any_numerical = any_numerical
        self.categorical = categorical
        self.numerical = numerical
        self.output = output
        self.categorical_embeddings = categorical_embeddings
        self.n_classes = n_classes

    @property
    def numerical_shape(self):
        if self.any_categorical:
            return self.numerical.shape[1]
        else:
            return 0

    def gain_all(self):
        return {'any_categorical': self.any_categorical, 'any_numerical': self.any_numerical, 'categorical': self.categorical, 'numerical': self.numerical, 'output': self.output, 'categorical_embeddings': self.categorical_embeddings, 'numerical_shape': self.numerical_shape, 'n_classes': self.n_classes}


class DataRoles:
    def __init__(self, train=None, validation=None, test=None, categorical_embeddings=[], numerical_shape=None, n_classes=None, non_sampled=None):

        if non_sampled is None:
            self.train = train
            self.validation = validation
            self.test = test
            self.categorical_embeddings = categorical_embeddings
            self.numerical_shape = numerical_shape
            self.n_classes = n_classes
        else:
            self.train, self.validation, self.test, self.categorical_embeddings, self.numerical_shape, self.n_classes = Samples.sample(**non_sampled.gain_all())

        self.any_categorical = self.train.any_categorical
        self.any_numerical = self.train.any_numerical


class Conductor:
    def __init__(self, data_frame, target, embedding_strategy='default', embedding_explicit=[]):
        self.data_frame = data_frame
        self._data = None
        if isinstance(target, list):
            self.target = target
        else:
            self.target = [target]
        self._embedding_strategy = embedding_strategy
        self._embedding_explicit = embedding_explicit

    @property
    def data(self):
        if self._data is None:
            self.data_cast()
        return self._data

    def data_cast(self):
        n_categorical = sum([(self.data_frame[col].dtype.name == 'category') and (col not in self.target) for col in self.data_frame.columns.values])
        n_numerical = sum([(self.data_frame[col].dtype.name == 'float64') and (col not in self.target) for col in self.data_frame.columns.values])
        any_categorical = n_categorical > 0
        any_numerical = n_numerical > 0
        data = DataFormats(any_categorical, any_numerical)

        if not any_categorical:
            data.categorical = None
        else:
            data.categorical = numpy.stack([self.data_frame[col].cat.codes.values for col in self.data_frame.columns.values if (self.data_frame[col].dtype.name == 'category') and (col not in self.target)], axis=1)
        if not any_numerical:
            data.numerical = None
        else:
            data.numerical = numpy.stack([self.data_frame[col].values for col in self.data_frame.columns.values if (self.data_frame[col].dtype.name == 'float64') and (col not in self.target)], axis=1)
        data.output = self.data_frame[self.target].values
        if self._embedding_strategy == 'default':
            data.categorical_embeddings = [(len(self.data_frame[col].cat.categories), min(50, (len(self.data_frame[col].cat.categories) + 1) // 2)) for col in self.data_frame.columns.values if (self.data_frame[col].dtype.name == 'category') and (col not in self.target)]
        elif self._embedding_strategy is None:
            data.categorical_embeddings = []
        else:
            data.categorical_embeddings = self._embedding_explicit
        if self.data_frame[self.target[0]].dtype.name == 'category':
            data.n_classes = self.data_frame[self.target[0]].cat.categories.values.shape[0]
        else:
            data.n_classes = None
        self._data = DataRoles(non_sampled=data)



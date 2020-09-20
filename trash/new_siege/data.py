#
import numpy


#
import torch
from torch import nn


#


class DataStore:

    def __init__(self, qualitative, qualitative_embeddings, quantitative, quantitative_embeddings, output):
        self.qualitative = qualitative
        self.qualitative_embeddings = qualitative_embeddings
        self.quantitative = quantitative
        self.quantitative_embeddings = quantitative_embeddings
        self.output = output


class DataTreat:

    def __init__(self, data_frame, target, qualitative_embeddings, quantitative_embeddings, any_qualitative, any_quantitative, test_rate=0.2, validation_rate=0.2):

        self.verbose = True

        self.test_rate = test_rate
        self.validation_rate = validation_rate

        self.any_qualitative = any_qualitative
        self.any_quantitative = any_quantitative

        if self.any_qualitative:
            self.qualitative = numpy.stack([data_frame[col].cat.codes.values for col in data_frame.columns.values if (data_frame[col].dtype.name == 'category') and (col not in target)], axis=1)
            if isinstance(qualitative_embeddings, list):
                self.qualitative_embeddings = qualitative_embeddings
            elif qualitative_embeddings == 'default':
                self.qualitative_embeddings = [(len(data_frame[col].cat.categories), min(50, (len(data_frame[col].cat.categories) + 1) // 2)) for col in data_frame.columns.values if (data_frame[col].dtype.name == 'category') and (col not in target)]
            else:
                raise Exception("Idk the kind of embeddings you supposed to be")
        else:
            self.qualitative = None
            self.qualitative_embeddings = None
        if self.any_quantitative:
            self.quantitative = numpy.stack([data_frame[col].values for col in data_frame.columns.values if (data_frame[col].dtype.name == 'float64') and (col not in target)], axis=1)
            self.quantitative_embeddings = quantitative_embeddings
        else:
            self.quantitative = None
            self.quantitative_embeddings = None
        self.output = data_frame[target].values
        # buggy
        self.output_dtype = data_frame[target[0]].dtype.name

    @property
    def samples(self):

        # Pytorch Data Preparation

        if self.any_qualitative:
            self.qualitative = torch.tensor(self.qualitative, dtype=torch.int64)
        if self.any_quantitative:
            self.quantitative = torch.tensor(self.quantitative, dtype=torch.float)
        if self.output_dtype == 'category':
            self.output = torch.tensor(self.output).flatten()
        elif self.output_dtype == 'float64':
            # self.output = torch.tensor(self.output, dtype=torch.float).flatten()
            self.output = torch.tensor(self.output, dtype=torch.float)
        else:
            raise Exception("Such a strange dtype of target variable...")
        if self.verbose:
            if self.any_qualitative:
                print(self.qualitative.shape)
            else:
                print(0)
            if self.any_quantitative:
                print(self.quantitative.shape)
            else:
                print(0)
            print(self.output.shape)

        # Train-Test Split

        samples = [int(self.test_rate * len(self.output))]
        samples = [len(self.output) - samples[0]] + samples
        a = numpy.array(numpy.arange(len(self.output)))
        a_train = numpy.random.choice(a, size=samples[0], replace=False)
        a_test = numpy.setdiff1d(a, a_train)

        if self.any_qualitative:
            qualitative_data_train, qualitative_data_test = self.qualitative[a_train], self.qualitative[a_test]
        else:
            qualitative_data_train, qualitative_data_test = None, None
        if self.any_quantitative:
            quantitative_data_train, quantitative_data_test = self.quantitative[a_train], self.quantitative[a_test]
        else:
            quantitative_data_train, quantitative_data_test = None, None
        output_data_train, output_data_test = self.output[a_train], self.output[a_test]

        if self.verbose:
            if self.any_qualitative:
                print(qualitative_data_train.shape, qualitative_data_test.shape)
            else:
                print(0, 0)
            if self.any_quantitative:
                print(quantitative_data_train.shape, quantitative_data_test.shape)
            else:
                print(0, 0)
            print(output_data_train.shape, output_data_test.shape)

        # Train-Validation Split

        samples = [int(self.validation_rate * len(output_data_train))]
        samples = [len(output_data_train) - samples[0]] + samples
        a = numpy.array(numpy.arange(len(output_data_train)))
        a_train = numpy.random.choice(a, size=samples[0], replace=False)
        a_validation = numpy.setdiff1d(a, a_train)

        if self.any_qualitative:
            qualitative_data_train, qualitative_data_validation = qualitative_data_train[a_train], qualitative_data_train[a_validation]
        else:
            qualitative_data_train, qualitative_data_validation = None, None
        if self.any_quantitative:
            quantitative_data_train, quantitative_data_validation = quantitative_data_train[a_train], quantitative_data_train[a_validation]
        else:
            quantitative_data_train, quantitative_data_validation = None, None
        output_data_train, output_data_validation = output_data_train[a_train], output_data_train[a_validation]

        if self.verbose:
            if self.any_qualitative:
                print(qualitative_data_train.shape, qualitative_data_validation.shape, qualitative_data_test.shape)
            else:
                print(0, 0)
            if self.any_quantitative:
                print(quantitative_data_train.shape, quantitative_data_validation.shape, quantitative_data_test.shape)
            else:
                print(0, 0)
            print(output_data_train.shape, output_data_validation.shape, output_data_test.shape)

        train = DataStore(qualitative_data_train, self.qualitative_embeddings, quantitative_data_train, self.quantitative_embeddings, output_data_train)
        validation = DataStore(qualitative_data_validation, self.qualitative_embeddings, quantitative_data_validation, self.quantitative_embeddings, output_data_validation)
        test = DataStore(qualitative_data_test, self.qualitative_embeddings, quantitative_data_test, self.quantitative_embeddings, output_data_test)

        return train, validation, test


class DataTypes:

    def __init__(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test

    @property
    def qualitative_embeddings(self):
        return self.train.qualitative_embeddings

    @property
    def quantitative_embeddings(self):
        return self.train.quantitative_embeddings

    @property
    def quantitative_d1(self):
        if self.train.quantitative is None:
            return 0
        else:
            return self.train.quantitative.shape[1]


class DataHandler:

    def __init__(self, data_frame, target, qualitative_embeddings='default', quantitative_embeddings=nn.BatchNorm1d):
        self.data_frame = data_frame
        if isinstance(target, list):
            self.target = target
        else:
            self.target = [target]
        self.qualitative_embeddings = qualitative_embeddings
        self.quantitative_embeddings = quantitative_embeddings
        self._data = None

    @property
    def data(self):
        if self._data is None:
            _data = DataTreat(self.data_frame, self.target, self.qualitative_embeddings, self.quantitative_embeddings, self.any_qualitative, self.any_quantitative)
            _train, _validation, _test = _data.samples
            self._data = DataTypes(_train, _validation, _test)
        return self._data

    @property
    def any_qualitative(self):
        return sum([(self.data_frame[col].dtype.name == 'category') and (col not in self.target) for col in self.data_frame.columns.values]) > 0

    @property
    def any_quantitative(self):
        return sum([(self.data_frame[col].dtype.name == 'float64') and (col not in self.target) for col in self.data_frame.columns.values]) > 0



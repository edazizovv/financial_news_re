
import torch
from torch import nn

import seaborn
import numpy
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot


class WrappedNumericOnlyGene:

    def __init__(self, numerical_shape,
                 layers, layers_dimensions,
                 activators,
                 optimiser, optimiser_kwargs, loss_function,
                 preprocessor, interprocessors,
                 interdrops,
                 epochs=500):

        self.optimiser = optimiser
        self.optimiser_kwargs = optimiser_kwargs
        self.loss_function = loss_function
        self.epochs = epochs

        self.model = NumericOnlyGene(numerical_shape=numerical_shape,
                 layers=layers, layers_dimensions=layers_dimensions,
                 activators=activators,
                 interprocessors=interprocessors, interdrops=interdrops,
                 preprocessor=preprocessor)

    def fit(self, X_train, y_train, X_val, y_val):

        _optimizer = self.optimiser(self.model.parameters(), **self.optimiser_kwargs)
        self.model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                       # optimiser=self.optimiser, optimiser_kwargs=self.optimiser_kwargs, loss_function=self.loss_function, epochs=self.epochs)
                       optimiser=_optimizer, loss_function=self.loss_function, epochs=self.epochs)

    def plot_fit(self):
        self.model.plot_fit()

    def predict(self, X):
        return self.model.predict(X)


class NumericOnlyGene(nn.Module):
    """
    Model architecture:

        -- Embedding
        -- EmbeddingDrop
        -- Preprocessor

        == LAYER X: -- Linear
                    -- Activator
                    -- Interprocessor (optional)
                    -- Interdrop      (optional)
    """

    def __init__(self, numerical_shape,
                 layers, layers_dimensions,
                 activators,
                 interprocessors, interdrops,
                 preprocessor,
                 econom=1
                 ):

        super().__init__()

        self.econom = econom
        # self.data = data
        self.numerical_shape = numerical_shape

        self.epochs = None
        self.aggregated_losses = None
        self.validation_losses = None

        # if data.data.categorical_embeddings is not None:

        if preprocessor is not None:
            self.batch_norm_num = preprocessor(numerical_shape)

        all_layers = []
        num_numerical_cols = numerical_shape
        input_size = num_numerical_cols

        for j in range(len(layers_dimensions)):
            all_layers.append(layers[j](input_size, layers_dimensions[j]))
            if activators[j] is not None:
                all_layers.append(activators[j]())
            if interprocessors[j] is not None:
                all_layers.append(interprocessors[j](layers_dimensions[j]))
            if interdrops[j] is not None:
                all_layers.append(nn.Dropout(interdrops[j]))
            input_size = layers_dimensions[j]

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_numerical):

        x = torch.cat([x_numerical], 1)
        x = self.layers(x)
        return x

    # def fit(self, X_train, y_train, X_val, y_val, optimiser, optimiser_kwargs, loss_function, epochs=500):
    def fit(self, X_train, y_train, X_val, y_val, optimiser, loss_function, epochs=500):

        self.epochs = epochs
        self.aggregated_losses = []
        self.validation_losses = []

        for i in range(epochs):
            i += 1
            for phase in ['train', 'validate']:

                if phase == 'train':
                    y_pred = self(X_train)
                    single_loss = loss_function(y_pred, y_train)
                else:
                    y_pred = self(X_val)
                    single_loss = loss_function(y_pred, y_val)

                optimiser.zero_grad()

                if phase == 'train':
                    train_lost = single_loss.item()
                    if self.econom == 0:
                        self.aggregated_losses.append(single_loss)
                    single_loss.backward()
                    optimiser.step()
                else:
                    validation_lost = single_loss.item()
                    if self.econom == 0:
                        self.validation_losses.append(single_loss)

            if i % 25 == 1:
                print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost,
                                                                                             validation_lost))
        print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost, validation_lost))

    def plot_fit(self):

        if self.econom == 0:
            pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.aggregated_losses, label='Train')
            pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.validation_losses, label='Validation')
            pyplot.legend(loc="upper left")
            pyplot.show()
        else:
            print('You ran training with econom mode, so no history was recorded')

    def predict(self, X_test):

        output = self(X_test)
        # result = numpy.argmax(output.detach().numpy(), axis=1)
        result = output.detach().numpy()

        return result


class WrappedGene:

    def __init__(self, categorical_embeddings, numerical_shape, n_classes,
                 layers, layers_dimensions,
                 activators,
                 optimiser, loss_function, epochs=500,
                 preprocessor=nn.BatchNorm1d, embeddingdrop=0.0, activators_args={}, interprocessors=None,
                 interdrops=None, postlayer=None):

        self.optimiser = optimiser
        self.loss_function = loss_function
        self.epochs = epochs

        self.model = Gene(categorical_embeddings, numerical_shape, n_classes,
                         layers, layers_dimensions,
                         activators,
                         preprocessor, embeddingdrop, activators_args, interprocessors,
                         interdrops, postlayer)

    def fit(self, X_train_categorical, X_train_numerical, y_train, X_val_categorical, X_val_numerical, y_val):

        self.model.fit(X_train_categorical, X_train_numerical, y_train, X_val_categorical, X_val_numerical, y_val,
                       optimiser=self.optimiser, loss_function=self.loss_function, epochs=self.epochs)

    def plot_fit(self):
        self.model.plot_fit()

    def predict(self, X_categorical, X_numerical):
        return self.model.predict(X_test_categorical=X_categorical, X_test_numerical=X_numerical)


class Gene(nn.Module):
    """
    Model architecture:

        -- Embedding
        -- EmbeddingDrop
        -- Preprocessor

        == LAYER X: -- Linear
                    -- Activator
                    -- Interprocessor (optional)
                    -- Interdrop      (optional)
    """

    def __init__(self, categorical_embeddings, numerical_shape, n_classes,
                 layers, layers_dimensions,
                 activators, preprocessor=nn.BatchNorm1d, embeddingdrop=0.0, activators_args={}, interprocessors=None,
                 interdrops=None, postlayer=None):
        super().__init__()

        # self.data = data
        self.categorical_embeddings = categorical_embeddings
        self.numerical_shape = numerical_shape
        self.n_classes = n_classes

        self.epochs = None
        self.aggregated_losses = None
        self.validation_losses = None

        # if data.data.categorical_embeddings is not None:
        if categorical_embeddings is not None:
            self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in categorical_embeddings])
            if preprocessor is not None:
                self.batch_norm_num = preprocessor(numerical_shape)
            self.embedding_dropout = nn.Dropout(embeddingdrop)
            num_categorical_cols = sum((nf for ni, nf in categorical_embeddings))
        else:
            self.all_embeddings = None
            num_categorical_cols = 0

        all_layers = []
        num_numerical_cols = numerical_shape
        input_size = num_categorical_cols + num_numerical_cols

        if not isinstance(activators, list):
            activators = [activators] * len(layers_dimensions)
        if not isinstance(activators_args, list):
            activators_args = [activators_args] * len(layers_dimensions)
        if interprocessors is not None and not isinstance(interprocessors, list):
            interprocessors = [interprocessors] * len(layers_dimensions)
        if interdrops is not None and not isinstance(interdrops, list):
            interdrops = [interdrops] * len(layers_dimensions)

        for j in range(len(layers_dimensions)):
            all_layers.append(layers[j](input_size, layers_dimensions[j]))
            all_layers.append(activators[j](**activators_args[j]))
            if interprocessors is not None:
                all_layers.append(interprocessors[j](layers_dimensions[j]))
            if interdrops is not None:
                all_layers.append(nn.Dropout(interdrops[j]))
            input_size = layers_dimensions[j]

        if postlayer is not None:
            all_layers.append(postlayer(layers_dimensions[-1], n_classes))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_numerical):
        if self.categorical_embeddings is not None:
            embeddings = []
            for i, e in enumerate(self.all_embeddings):
                embeddings.append(e(x_categorical[:, i]))
            x_embedding = torch.cat(embeddings, 1)
            x_embedding = self.embedding_dropout(x_embedding)
        else:
            x_embedding = None

        if self.categorical_embeddings is not None and self.numerical_shape is not None:
            x = torch.cat([x_embedding, x_numerical], 1)
        if self.categorical_embeddings is None and self.numerical_shape is not None:
            x = torch.cat([x_numerical], 1)
        if self.categorical_embeddings is not None and self.numerical_shape is None:
            x = torch.cat([x_embedding], 1)

        x = self.layers(x)
        return x

    def fit(self, X_train_categorical, X_train_numerical, y_train, X_val_categorical, X_val_numerical, y_val, optimiser, loss_function, epochs=500):

        self.epochs = epochs
        self.aggregated_losses = []
        self.validation_losses = []

        for i in range(epochs):
            i += 1
            for phase in ['train', 'validate']:

                if phase == 'train':
                    y_pred = self(X_train_categorical, X_train_numerical)
                    single_loss = loss_function(y_pred, y_train)
                else:
                    y_pred = self(X_val_categorical, X_val_numerical)
                    single_loss = loss_function(y_pred, y_val)

                optimiser.zero_grad()

                if phase == 'train':
                    train_lost = single_loss.item()
                    self.aggregated_losses.append(single_loss)
                    single_loss.backward()
                    optimiser.step()
                else:
                    validation_lost = single_loss.item()
                    self.validation_losses.append(single_loss)

            if i % 25 == 1:
                print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost,
                                                                                             validation_lost))
        print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost, validation_lost))

    def plot_fit(self):

        pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.aggregated_losses, label='Train')
        pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.validation_losses, label='Validation')
        pyplot.legend(loc="upper left")
        pyplot.show()

    def predict(self, X_test_categorical, X_test_numerical):

        output = self(X_test_categorical, X_test_numerical)
        result = numpy.argmax(output.detach().numpy(), axis=1)

        return result

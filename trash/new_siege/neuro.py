#
import torch
from torch import nn

#
import seaborn
import numpy
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot


#


class Gene(nn.Module):

    def __init__(self, data,
                       layers, layers_dimensions,
                       activators, activators_args, preprocessors,
                       embeddingdrop, drops):

        super().__init__()

        self.data = data

        self.epochs = None
        self.aggregated_losses = None
        self.validation_losses = None

        if self.data.data.qualitative_embeddings is not None:
            self.qualitative_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in data.data.qualitative_embeddings])
            num_categorical_cols = sum((nf for ni, nf in data.data.qualitative_embeddings))
            if embeddingdrop is not None:
                self.embedding_dropout = nn.Dropout(embeddingdrop)
            else:
                self.embedding_dropout = nn.Dropout(0)
        else:
            num_categorical_cols = 0
        if self.data.data.quantitative_embeddings is not None:
            self.quantitative_embeddings = self.data.data.quantitative_embeddings
            num_numerical_cols = data.data.quantitative_d1
        else:
            num_numerical_cols = 0
        input_size = num_categorical_cols + num_numerical_cols

        if not isinstance(activators, list):
            activators = [activators] * len(layers_dimensions)
        if not isinstance(activators_args, list):
            activators_args = [activators_args] * len(layers_dimensions)
        if not isinstance(preprocessors, list):
            preprocessors = [preprocessors] * len(layers_dimensions)
        if not isinstance(drops, list):
            drops = [drops] * len(layers_dimensions)

        all_layers = []
        for j in range(len(layers_dimensions)):
            all_layers.append(layers[j](input_size, layers_dimensions[j]))
            all_layers.append(activators[j](**activators_args[j]))
            if preprocessors[j] is not None:
                all_layers.append(preprocessors[j](layers_dimensions[j]))
            if drops[j] is not None:
                all_layers.append(nn.Dropout(drops[j]))
            input_size = layers_dimensions[j]

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_numerical):

        if self.data.data.qualitative_embeddings is not None:
            q_embeddings = []
            for i, e in enumerate(self.qualitative_embeddings):
                q_embeddings.append(e(x_categorical[:, i]))
            x_embedding = torch.cat(q_embeddings, 1)
            x_embedding = self.embedding_dropout(x_embedding)
        else:
            x_embedding = None

        if self.data.any_qualitative and self.data.any_quantitative:
            x = torch.cat([x_embedding, x_numerical], 1)
        elif not self.data.any_qualitative and self.data.any_quantitative:
            x = torch.cat([x_numerical], 1)
        elif self.data.any_qualitative and not self.data.any_quantitative:
            x = torch.cat([x_embedding], 1)
        else:
            raise Exception("No data?")

        x = self.layers(x)
        return x

    def fit(self, optimiser, loss_function, epochs=500):

        self.epochs = epochs
        self.aggregated_losses = []
        self.validation_losses = []

        for i in range(epochs):
            i += 1
            for phase in ['train', 'validate']:

                if phase == 'train':
                    y_pred = self(self.data.data.train.qualitative, self.data.data.train.quantitative)
                    """
                    print(y_pred.shape)
                    print(y_pred)
                    print(self.data.data.train.output.shape)
                    print(self.data.data.train.output)
                    """
                    single_loss = loss_function(y_pred, self.data.data.train.output)
                else:
                    y_pred = self(self.data.data.validation.qualitative, self.data.data.validation.quantitative)
                    single_loss = loss_function(y_pred, self.data.data.validation.output)

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

    def fit_plot(self):

        pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.aggregated_losses, label='Train')
        pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.validation_losses, label='Validation')
        pyplot.legend(loc="upper left")
        pyplot.show()

    def predict(self):

        output = self(self.data.data.test.qualitative, self.data.data.test.quantitative)
        result = numpy.argmax(output.detach().numpy(), axis=1)

        return result

    def summary(self, on='test', loss_function=None, show_confusion_matrix=True,
                report=False, score=None):

        if on == 'train':

            y_val = self(self.data.data.train.qualitative, self.data.data.train.quantitative)
            y_hat = self.predict()
            y = self.data.data.test.output.detach().numpy()

            if loss_function is not None:
                print('{0:25}: {1:10.8f}'.format(str(loss_function)[:-2], loss_function(y_val, self.data.data.train.output)))

        if on == 'validation':

            y_val = self(self.data.data.validation.qualitative, self.data.data.validation.quantitative)
            y_hat = self.predict()
            y = self.data.data.validation.output.detach().numpy()

            if loss_function is not None:
                print('{0:25}: {1:10.8f}'.format(str(loss_function)[:-2], loss_function(y_val, self.data.data.validation.output)))

        if on == 'test':

            y_val = self(self.data.data.test.qualitative, self.data.data.test.quantitative)
            y_hat = self.predict()
            y = self.data.data.test.output.detach().numpy()

            if loss_function is not None:
                print('{0:25}: {1:10.8f}'.format(str(loss_function)[:-2], loss_function(y_val, self.data.data.test.output)))

        if show_confusion_matrix:
            seaborn.heatmap(confusion_matrix(y, y_hat), annot=True)

        if report:
            print(classification_report(y, y_hat))

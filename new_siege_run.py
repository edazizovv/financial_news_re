#
import numpy
import pandas


#
import torch
from torch import nn


#


N = 10_000
data_raw = pandas.DataFrame(data={'A': numpy.array(numpy.arange(N)),
# 'A': numpy.random.choice([0, 1], size=(N,)),
                              'B': numpy.array(numpy.arange(N)) + numpy.random.normal(size=(N,)),
                              'C': numpy.random.normal(size=(N,))})

target = 'A'
quantitative = ['B', 'C']
qualitative = []

# decise

data_set = data_raw.copy()

for category in qualitative:
    data_set[category] = data_set[category].astype('category')
for numeric in quantitative:
    data_set[numeric] = data_set[numeric].astype('float64')

data_set[target] = data_set[target].astype('float64')
# data_set[target] = data_set[target].astype('category')

from new_siege.data import DataHandler

data = DataHandler(data_set, target)
print(data.data)

from new_siege.neuro import Gene

# model-params

layers = [nn.Linear, nn.Linear]
layers_dimensions = [100, 1]
# layers_dimensions = [100, 2]
# activators = [nn.Sigmoid, nn.Sigmoid]
activators = [nn.ReLU, nn.ReLU]
activators_args = {}
# preprocessors = nn.BatchNorm1d
preprocessors = None
embeddingdrop = 0.0
drops = 0.0

model = Gene(data, layers, layers_dimensions, activators, activators_args, preprocessors, embeddingdrop, drops)
print(model)

# Define Optimisation

loss_function = nn.MSELoss()
# loss_function = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the Model

epochs = 500
model.fit(optimiser, loss_function, epochs)

model.fit_plot()

model.summary(loss_function=loss_function, show_confusion_matrix=False)


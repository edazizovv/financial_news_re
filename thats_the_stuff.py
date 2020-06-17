#
import numpy
import pandas
import datetime

from m_utils.measures import r2_adj
from m_utils.sampling import ts_sampler


def load_data(data, y_names, removes, test_rate, n_folds):
    # clarify names

    exclude = y_names + removes
    x_names = [x for x in data.columns.values if (x not in exclude and 'LAG0' not in x)]

    # sample (without folds)

    data_train, data_test = ts_sampler(data, n_folds, test_rate)
    X_, Y_ = numpy.array(x_names), numpy.array(y_names)
    X_train, Y_train = [x[X_].values for x in data_train], [x[Y_].values for x in data_train]
    X_test, Y_test = data_test[X_].values, data_test[Y_].values

    print('Fold shape: {0}'.format(X_train[0].shape))
    print('Test shape: {0}'.format(X_test.shape))

    return X_train, Y_train, X_test, Y_test, X_


def stop_out(b):
    b[numpy.isnan(b)] = 0
    b = b.astype(dtype=numpy.float32)
    b[b == numpy.inf] = numpy.finfo(numpy.float32).max
    b[b == -numpy.inf] = numpy.finfo(numpy.float32).min
    return b


class Distributed:

    def __init__(self, my_name):
        self.my_name = my_name
        self.store = None

    def say_my_name(self):
        return self.my_name

    def fit(self, array):

        if self.my_name == 'Nothing':

            pass

        elif self.my_name == 'Simple':

            self.store = {'mean': array.mean(), 'std': array.std(ddof=1)}

        elif self.my_name == 'Normal':

            # from sklearn.preprocessing import PowerTransformer
            # self.store = PowerTransformer()
            from sklearn.preprocessing import QuantileTransformer
            self.store = QuantileTransformer(output_distribution='normal')
            arr = array.copy().astype(dtype=numpy.float64)
            self.store.fit(arr)

        elif self.my_name == 'Uniform':

            # from sklearn.preprocessing import PowerTransformer
            # self.store = PowerTransformer()
            from sklearn.preprocessing import QuantileTransformer
            self.store = QuantileTransformer(output_distribution='uniform')
            arr = array.copy().astype(dtype=numpy.float64)
            self.store.fit(arr)

        else:

            raise Exception("Not Yet!")

    def forward(self, array):

        if self.my_name == 'Nothing':

            arr = array.copy()

            return arr

        elif self.my_name == 'Simple':

            arr = array.copy().astype(dtype=numpy.float64)
            arr = (arr - self.store['mean']) / self.store['std']
            arr = stop_out(arr)

            return arr

        elif self.my_name == 'Normal':

            arr = array.copy().astype(dtype=numpy.float64)
            arr = self.store.transform(arr)
            arr = stop_out(arr)

            return arr

        elif self.my_name == 'Uniform':

            arr = array.copy().astype(dtype=numpy.float64)
            arr = self.store.transform(arr)
            arr = stop_out(arr)

            return arr

        else:

            raise Exception("Not Yet!")

    def backward(self, array):

        if self.my_name == 'Nothing':

            arr = array.copy()

            return arr

        elif self.my_name == 'Simple':

            arr = array.copy().astype(dtype=numpy.float64)
            arr = arr * self.store['std'] + self.store['mean']
            arr = stop_out(arr)

            return arr

        elif self.my_name == 'Normal':

            arr = array.copy().astype(dtype=numpy.float64)
            arr = self.store.inverse_transform(arr)
            arr = stop_out(arr)

            return arr

        elif self.my_name == 'Uniform':

            arr = array.copy().astype(dtype=numpy.float64)
            arr = self.store.inverse_transform(arr)
            arr = stop_out(arr)

            return arr

        else:

            raise Exception("Not Yet!")


def goded(das_model, data, multiple_model_args, tsi_names, y_names, removes, test_rate=0.2, n_folds=1):
    report = pandas.DataFrame(
        columns=['Np', 'Nf', 'Ns', 'R2_adj_cur_fold', 'R2_adj_nxt_fold', 'R2_adj_test', 'smoother', 'd1', 'params',
                 'd1', 'X_adj_'])
    X_train, Y_train, X_test, Y_test, X_ = load_data(data, y_names, removes, test_rate, n_folds)

    # smoothers = [Distributed, Distributed, Distributed, Distributed]
    smoothers = [Distributed, Distributed, Distributed]
    # smoothers_args = [{'my_name': 'Nothing'}, {'my_name': 'Simple'}, {'my_name': 'Normal'}, {'my_name': 'Uniform'}]
    smoothers_args = [{'my_name': 'Nothing'}, {'my_name': 'Simple'}, {'my_name': 'Normal'}]

    verbose_step = 100
    n_iters = len(smoothers) * len(multiple_model_args)
    print('N of expected iters = {0}'.format(n_iters))
    print('Started search: {0}'.format(datetime.datetime.now().isoformat()))

    it = 0

    for s in range(len(smoothers)):

        smoother_X_train = []
        smoother_Y_train = []
        for j in range(len(X_train)):
            smt_X = smoothers[s](**smoothers_args[s])
            smt_X.fit(array=X_train[j])
            smoother_X_train.append(smt_X)

            smt_Y = smoothers[s](**smoothers_args[s])
            smt_Y.fit(array=Y_train[j])
            smoother_Y_train.append(smt_Y)

        X_train_ = [smoother_X_train[z].forward(array=X_train[z]) for z in range(len(X_train))]
        Y_train_ = [smoother_Y_train[z].forward(array=Y_train[z]) for z in range(len(Y_train))]

        for i in range(len(multiple_model_args)):

            if it % verbose_step == 0:
                print('{0} / {1}'.format(it, n_iters))

            params = multiple_model_args[i]

            for j in range(len(X_train_)):

                model_ = das_model(**params)
                model_.fit(X_train_[j], Y_train_[j].ravel())

                Y_hat_train = smoother_Y_train[j].backward(array=model_.predict(X_train_[j]).reshape(-1, 1))
                Y_hat_test = smoother_Y_train[j].backward(
                    array=model_.predict(smoother_X_train[j].forward(array=X_test)).reshape(-1, 1))

                if j < (len(X_train_) - 1):

                    Y_hat_ded = smoother_Y_train[j].backward(
                        array=model_.predict(smoother_X_train[j].forward(array=X_train_[(j + 1)])).reshape(-1, 1))
                    nxt_folded = r2_adj(Y_train[(j + 1)], Y_hat_ded, X_train_[j].shape[0], X_train_[j].shape[1])

                else:

                    nxt_folded = r2_adj(Y_test, Y_hat_test, X_train_[j].shape[0], X_train_[j].shape[1])

                result = {'Np': i, 'Nf': j, 'Ns': s,
                          'R2_adj_cur_fold': r2_adj(Y_train[j], Y_hat_train, X_train_[j].shape[0],
                                                    X_train_[j].shape[1]),
                          'R2_adj_nxt_fold': nxt_folded,
                          'R2_adj_test': r2_adj(Y_test, Y_hat_test, X_train_[j].shape[0], X_train_[j].shape[1]),
                          'smoother': smoother_X_train[j].say_my_name(),
                          'd1': X_train_[j].shape[1], 'params': params, 'X_adj_': X_}

                report = report.append(result, ignore_index=True)

            it += 1

    print('{0} / {0}'.format(n_iters))
    print('Finished search: {0}'.format(datetime.datetime.now().isoformat()))

    return report

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


def golags(das_model, data, multiple_model_args, tsi_names, y_names, removes, test_rate=0.2, n_folds=1):
    report = pandas.DataFrame(
        columns=['Np', 'Nf', 'R2_adj_cur_fold', 'R2_adj_nxt_fold', 'R2_adj_test', 'd1', 'params', 'd1', 'X_adj_'])
    X_train, Y_train, X_test, Y_test, X_ = load_data(data, y_names, removes, test_rate, n_folds)

    verbose_step = 100
    n_iters = len(multiple_model_args)
    print('N of expected iters = {0}'.format(n_iters))
    print('Started search: {0}'.format(datetime.datetime.now().isoformat()))

    for i in range(len(multiple_model_args)):

        if i % verbose_step == 0:
            print('{0} / {1}'.format(i, n_iters))

        params = multiple_model_args[i]

        for j in range(len(X_train)):

            model_ = das_model(**params)
            model_.fit(X_train[j], Y_train[j].ravel())
            Y_hat_train = model_.predict(X_train[j])
            Y_hat_test = model_.predict(X_test)

            if j < (len(X_train) - 1):

                Y_hat_ded = model_.predict(X_train[(j + 1)])
                nxt_folded = r2_adj(Y_train[(j + 1)], Y_hat_ded, X_train[(j + 1)].shape[0], X_train[(j + 1)].shape[1])

            else:

                nxt_folded = r2_adj(Y_test, Y_hat_test, X_test.shape[0], X_test.shape[1])

            result = {'Np': i, 'Nf': j,
                      'R2_adj_cur_fold': r2_adj(Y_train[j], Y_hat_train, X_train[j].shape[0], X_train[j].shape[1]),
                      'R2_adj_nxt_fold': nxt_folded,
                      'R2_adj_test': r2_adj(Y_test, Y_hat_test, X_test.shape[0], X_test.shape[1]),
                      'd1': X_train[j].shape[1], 'params': params, 'X_adj_': X_}

            report = report.append(result, ignore_index=True)

    print('{0} / {0}'.format(n_iters))
    print('Finished search: {0}'.format(datetime.datetime.now().isoformat()))

    return report

#
import pandas
import datetime

from m_utils.measures import r2_adj

from thats_the_stuff import load_data

#
from m_utils.transformations import LogPctTransformer, Whitener, HypeTan  # , Axe  <-- coming soon


# the following is actual


class Insane:

    def __init__(self, my_name):
        self.my_name = my_name
        self.store = None

    def say_my_name(self):
        return self.my_name

    def fit(self, array):

        if self.my_name == 'Nothing':

            pass

        elif self.my_name == 'LnPct':

            trf = LogPctTransformer()
            trf.fit(array)
            self.store = trf

        elif self.my_name == 'TanhLnPct':

            trf0 = LogPctTransformer()
            trf0.fit(array)
            array_ = trf0.transform(array)
            trf1 = HypeTan()
            trf1.fit(array_)
            trf = [trf0, trf1]
            self.store = trf

        elif self.my_name == 'Whiten':

            trf = Whitener()
            trf.fit(array)
            self.store = trf

        elif self.my_name == 'TanhWhiten':

            trf0 = Whitener()
            trf0.fit(array)
            array_ = trf0.transform(array)
            trf1 = HypeTan()
            trf1.fit(array_)
            trf = [trf0, trf1]
            self.store = trf

        elif self.my_name == 'AxeLnPct':

            """
            trf0 = LogPctTransformer()
            trf0.fit(array)
            array_ = trf0.tranform(array)
            trf1 = Axe()
            trf1.fit(array_)
            """

            raise Exception("It is coming soon...")

        elif self.my_name == 'AxeWOELnPct':

            raise Exception("It is coming soon...")

        else:

            raise Exception("Not Yet!")

    def forward(self, array):

        if self.my_name == 'Nothing':

            return array

        elif self.my_name == 'LnPct':

            return self.store.transform(array)

        elif self.my_name == 'TanhLnPct':

            return self.store[1].transform(self.store[0].transform(array))

        elif self.my_name == 'Whiten':

            return self.store.transform(array)

        elif self.my_name == 'TanhWhiten':

            return self.store[1].transform(self.store[0].transform(array))

        elif self.my_name == 'AxeLnPct':

            # return self.store[1].transform(self.store[0].transform(array))

            raise Exception("It is coming soon...")

        elif self.my_name == 'AxeWOELnPct':

            raise Exception("It is coming soon...")

        else:

            raise Exception("Not Yet!")

    def backward(self, array):

        if self.my_name == 'Nothing':

            pass

        elif self.my_name == 'LnPct':

            return self.store.inverse_transform(array)

        elif self.my_name == 'TanhLnPct':

            return self.store[0].inverse_transform(self.store[1].inverse_transform(array))

        elif self.my_name == 'Whiten':

            return self.store.inverse_transform(array)

        elif self.my_name == 'TanhWhiten':

            return self.store[0].inverse_transform(self.store[1].inverse_transform(array))

        elif self.my_name == 'AxeLnPct':

            # return self.store[0].inverse_transform(self.store[1].inverse_transform(array))

            raise Exception("It is coming soon...")

        elif self.my_name == 'AxeWOELnPct':

            raise Exception("It is coming soon...")

        else:

            raise Exception("Not Yet!")


# RENAME: transformers --> transformators !! there shall be no term confusions

class Compakt:

    def __init__(self, masked, coded):
        self.masked = masked
        self.coded = coded
        self.transformers = list(self.masked.keys())
        self.masks = [self.masked[key] for key in self.transformers]
        self.n = len(self.transformers)

    def say_my_name(self):

        return self.coded

    def fit(self, array):

        for j in range(self.n):
            self.transformers[j].fit(array[self.masks[j]])

    def forward(self, array):

        array_ = array.copy()

        for j in range(self.n):
            array_ = self.transformers[j].transform(array_[self.masks[j]])

        return array_

    def backward(self, array):

        array_ = array.copy()

        for j in range(self.n):
            array_ = self.transformers[-j - 1].inverse_transform(array_[self.masks[-j - 1]])

        return array_


def insane(das_model, data, quotes_mask, news_mask, target_mask, multiple_model_args, tsi_names, y_names, removes, test_rate=0.2, n_folds=1):
    report = pandas.DataFrame(
        columns=['Np', 'Nf', 'Ns', 'R2_adj_cur_fold', 'R2_adj_nxt_fold', 'R2_adj_test', 'smoother', 'd1', 'params',
                 'd1', 'X_adj_'])
    X_train, Y_train, X_test, Y_test, X_ = load_data(data, y_names, removes, test_rate, n_folds)

    maskeds_X = [{Insane(my_name='LnPct'): quotes_mask, Insane(my_name='Nothing'): news_mask},
               {Insane(my_name='TanhLnPct'): quotes_mask, Insane(my_name='Nothing'): news_mask},
               {Insane(my_name='LnPct'): quotes_mask, Insane(my_name='LnPct'): news_mask},
               {Insane(my_name='TanhLnPct'): quotes_mask, Insane(my_name='TanhLnPct'): news_mask},
               {Insane(my_name='Whiten'): quotes_mask, Insane(my_name='Nothing'): news_mask},
               {Insane(my_name='TanhWhiten'): quotes_mask, Insane(my_name='Nothing'): news_mask},
               {Insane(my_name='Whiten'): quotes_mask, Insane(my_name='Whiten'): news_mask},
               {Insane(my_name='TanhWhiten'): quotes_mask, Insane(my_name='TanhWhiten'): news_mask}]

    maskeds_Y = [{Insane(my_name='LnPct'): target_mask},
                 {Insane(my_name='TanhLnPct'): target_mask},
                 {Insane(my_name='LnPct'): target_mask},
                 {Insane(my_name='TanhLnPct'): target_mask},
                 {Insane(my_name='Whiten'): target_mask},
                 {Insane(my_name='TanhWhiten'): target_mask},
                 {Insane(my_name='Whiten'): target_mask},
                 {Insane(my_name='TanhWhiten'): target_mask}]

    maskeds_coded = ['LnPct_No', 'TanhLnPct_No', 'LnPct_LnPct', 'TanhLnPct_TanhLnPct',
                     'Whiten_No', 'TanhWhiten_No', 'Whiten_Whiten', 'TanhWhiten_TanhWhiten']

    verbose_step = 10
    n_iters = len(maskeds_X) * len(multiple_model_args) * n_folds
    print('N of expected iters = {0}'.format(n_iters))
    print('Started search: {0}'.format(datetime.datetime.now().isoformat()))

    it = 0

    for s in range(len(maskeds_X)):

        smoother_X_train = []
        smoother_Y_train = []

        # ????
        for j in range(len(X_train)):
            smt_X = Compakt(maskeds_X[j], maskeds_coded[j])
            smt_X.fit(array=X_train[j])
            smoother_X_train.append(smt_X)

            smt_Y = Compakt(maskeds_Y[j], maskeds_coded[j])
            smt_Y.fit(array=Y_train[j])
            smoother_Y_train.append(smt_Y)

        X_train_ = [smoother_X_train[z].forward(array=X_train[z]) for z in range(len(X_train))]
        Y_train_ = [smoother_Y_train[z].forward(array=Y_train[z]) for z in range(len(Y_train))]

        for i in range(len(multiple_model_args)):

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

                if it % verbose_step == 0:
                    print('{0} / {1}'.format(it, n_iters))

                it += 1

    print('{0} / {0}'.format(n_iters))
    print('Finished search: {0}'.format(datetime.datetime.now().isoformat()))

    return report

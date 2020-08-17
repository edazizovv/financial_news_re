#
import numpy
import pandas


#
from mpydge.wrap.new_data import DataHandler, MaskHandler
from mpydge.chaotic.the_new_pipe import SimplePipe
from supp import Insane
from supp import MAE, R2_adj


#
def glaive_thrower(code, models, transformators, data, y_name, removes, news_columns, quotes_columns, qualitative, quantitative, X_names, Y_names, output_spec, rfe_cv):
    report = pandas.DataFrame(
        columns=['it', 'code', 'transformator', 'model', 'rfe_cv', 'MAE_train', 'MAE_test', 'R2adj_train', 'R2adj_test'])
    plots = {}

    data_ = data.copy()
    data_ = data_.drop(columns=removes)

    quotes_mask = numpy.array([x in quotes_columns for x in data_.columns.values])
    news_mask = numpy.array([x in news_columns for x in data_.columns.values])
    target_mask = numpy.array([x == y_name for x in data_.columns.values])

    maskeds = [{Insane(my_name='LnPct'): quotes_mask, Insane(my_name='Nothing'): news_mask, Insane(my_name='LnPct'): target_mask},
               {Insane(my_name='TanhLnPct'): quotes_mask, Insane(my_name='Nothing'): news_mask, Insane(my_name='TanhLnPct'): target_mask},
               {Insane(my_name='LnPct'): quotes_mask, Insane(my_name='LnPct'): news_mask, Insane(my_name='LnPct'): target_mask},
               {Insane(my_name='TanhLnPct'): quotes_mask, Insane(my_name='TanhLnPct'): news_mask, Insane(my_name='TanhLnPct'): target_mask},
               {Insane(my_name='Whiten'): quotes_mask, Insane(my_name='Nothing'): news_mask, Insane(my_name='Whiten'): target_mask},
               {Insane(my_name='TanhWhiten'): quotes_mask, Insane(my_name='Nothing'): news_mask, Insane(my_name='TanhWhiten'): target_mask},
               {Insane(my_name='Whiten'): quotes_mask, Insane(my_name='Whiten'): news_mask, Insane(my_name='Whiten'): target_mask},
               {Insane(my_name='TanhWhiten'): quotes_mask, Insane(my_name='TanhWhiten'): news_mask, Insane(my_name='TanhWhiten'): target_mask}]

    maskeds_coded = ['LnPct_No', 'TanhLnPct_No', 'LnPct_LnPct', 'TanhLnPct_TanhLnPct',
                     'Whiten_No', 'TanhWhiten_No', 'Whiten_Whiten', 'TanhWhiten_TanhWhiten']

    it = 0

    for s in range(len(maskeds_coded)):

        for model in models:

            for transformator in transformators:

                thresh = int(data_.shape[0] * 0.8)
                train_mask = [x < thresh for x in range(data_.shape[0])]
                validation_mask = None
                test_mask = [x >= thresh for x in range(data_.shape[0])]
                sample_mask = MaskHandler(train=train_mask, validation=validation_mask, test=test_mask)

                qualitative_ = [x for x in qualitative if x in data_.columns.values]
                quantitative_ = [x for x in quantitative if x in data_.columns.values]

                dta = DataHandler(data_frame=data_, qualitative=qualitative_, quantitative=quantitative_, sample_mask=sample_mask)
                dta.sample()

                items = [transformator, model]
                items_args = [{'masked': maskeds[s], 'coded': maskeds_coded[s]}, {'rfe_cv': rfe_cv}]

                X_names = X_names
                Y_names = Y_names
                output_spec = output_spec

                pipe = SimplePipe(data=dta, items=items, items_args=items_args,
                                  X_names=X_names, Y_names=Y_names, output_spec=output_spec)

                pipe.fit()
                pipe_on_train = pipe.infer(on='train')
                pipe_on_test = pipe.infer(on='test')

                reversed_train = pipe.the_pipe[0].backward(pipe_on_train.train[list(output_spec[0].keys())].values)
                reversed_test = pipe.the_pipe[0].backward(pipe_on_test.test[list(output_spec[0].keys())].values)

                mae_train = MAE(y_true=dta.train[y_name].values, y_pred=reversed_train[:, -1])
                r2_adj_train = R2_adj(y_true=dta.train[y_name].values, y_pred=reversed_train[:, -1],
                                          dim1=len(list(output_spec[0].keys())))
                mae_test = MAE(y_true=dta.test[y_name].values, y_pred=reversed_test[:, -1])
                r2_adj_test = R2_adj(y_true=dta.test[y_name].values, y_pred=reversed_test[:, -1],
                                         dim1=len(list(output_spec[0].keys())))

                result = {'it': it, 'code': code,
                          'transformator': transformator, 'model': model,
                          'rfe_cv': rfe_cv,
                          'MAE_train': mae_train, 'MAE_test': mae_test,
                          'R2adj_train': r2_adj_train, 'R2adj_test': r2_adj_test}
                report = report.append(result, ignore_index=True)

                y_true_train = dta.train[y_name].values
                y_true_test = dta.test[y_name].values
                y_hat_train = reversed_train[:, -1]
                y_hat_test = reversed_test[:, -1]

                plots[it] = {'y_true_train': y_true_train,
                             'y_true_test': y_true_test,
                             'y_hat_train': y_hat_train,
                             'y_hat_test': y_hat_test}

                # pyplot.plot(range(1000), ys_true_train[0], 'navy', range(1000), ys_hat_train[0], 'orange')
                # pyplot.plot(range(227), ys_true_test[0], 'navy', range(227), ys_hat_test[0], 'orange')

                it += 1

    return report

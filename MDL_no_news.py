#
import pytz
import pandas
import datetime
from matplotlib import pyplot

#
from m_utils.transform import lag_it
from mpydge.wrap.new_data import DataHandler, MaskHandler
from mpydge.chaotic.the_new_pipe import SimplePipe
from new_insane import Insane, Neakt
from new_insane import XBR
from new_insane import MAE, R2_adj
from reporter import reported_n_muted


print("Process Started:\t {0}".format(datetime.datetime.utcnow().replace(tzinfo=pytz.utc)))
#
d = './result.csv'

dataset = pandas.read_csv(d, sep=';')

dataset = dataset.set_index(['time'])
# dataset = dataset.sort_index(ascending=True)

# dataset = lag_it(dataset, n_lags=1, exactly=False)
dataset = dataset.dropna()

dogtag = dataset.copy()
drops = ['id', 'title', 'news_time'] + ['ticker']  # ticker should be embedded!
dogtag = dogtag.drop(columns=drops)
cols_to_drop = [x for x in dogtag.columns.values if ('LAG0' in x and 'close' not in x)]
dogtag = dogtag.drop(columns=cols_to_drop)

target = 'close_LAG0'
qualitative = []
quantitative = [x for x in dogtag.columns.values if 'LAG0' not in x]

X_names = [qualitative + quantitative + [target], qualitative + quantitative]
Y_names = [target, target]
outputs_0 = qualitative + quantitative + [target]
output_spec = [{x: 'float64' for x in outputs_0}, {target: 'float64'}]

g_mask = [x in quantitative for x in X_names[0]]  # !
target_mask = [x == target for x in X_names[0]]

# they are similar both for target and factors, remove redundant, pls
maskeds = [{Insane(my_name='Nothing'): g_mask, Insane(my_name='Nothing'): target_mask},
           {Insane(my_name='TanhLnPct'): g_mask, Insane(my_name='TanhLnPct'): target_mask},
           {Insane(my_name='LnPct'): g_mask, Insane(my_name='LnPct'): target_mask},
           {Insane(my_name='Whiten'): g_mask, Insane(my_name='Whiten'): target_mask},
           {Insane(my_name='TanhWhiten'): g_mask, Insane(my_name='TanhWhiten'): target_mask}]

maskeds_coded = ['Nothing', 'TanhLnPct', 'LnPct', 'Whiten', 'TanhWhiten']

train_rate = 0.7
full_len = dogtag.shape[0]
train_len = int(full_len * train_rate)
test_len = full_len - train_len

train_mask = [x < train_len for x in range(dogtag.shape[0])]
test_mask = [x >= train_len for x in range(dogtag.shape[0])]
sample_mask = MaskHandler(train=train_mask, validation=None, test=test_mask)
dta = DataHandler(data_frame=dogtag, qualitative=qualitative, quantitative=quantitative, sample_mask=sample_mask)
dta.sample()

ys_true_train = []
ys_true_test = []
ys_hat_train = []
ys_hat_test = []

maes_train, r2_js_train = [], []
maes_test, r2_js_test = [], []

report_measures = pandas.DataFrame(columns=['iteration', 'fold', 'measure', 'value'])
report_values = pandas.DataFrame(columns=['iteration', 'fold', 'tix', 'y_true', 'y_hat'])

models = [XBR]
models_names = ['XBR']

for i in range(len(models_names)):

    items = [Neakt, models[i]]
    for j in range(len(maskeds_coded)):

        items_args = [{'masked': maskeds[j], 'coded': maskeds_coded[j]}, {'rfe_cv': True}]
        pipe = SimplePipe(data=dta, items=items, items_args=items_args,
                          X_names=X_names, Y_names=Y_names, output_spec=output_spec)

        pipe.fit()
        pipe_on_train = pipe.infer(on='train')
        pipe_on_test = pipe.infer(on='test')

        reversed_train = pipe.the_pipe[0].backward(pipe_on_train.train[list(output_spec[0].keys())].values)
        reversed_test = pipe.the_pipe[0].backward(pipe_on_test.test[list(output_spec[0].keys())].values)

        ys_true_train.append(dta.train[target].values)
        ys_true_test.append(dta.test[target].values)
        ys_hat_train.append(reversed_train[:, -1])
        ys_hat_test.append(reversed_test[:, -1])

        maes_train.append(MAE(y_true=dta.train[target].values, y_pred=reversed_train[:, -1]))
        r2_js_train.append(R2_adj(y_true=dta.train[target].values, y_pred=reversed_train[:, -1], dim1=len(list(output_spec[0].keys()))))
        maes_test.append(MAE(y_true=dta.test[target].values, y_pred=reversed_test[:, -1]))
        r2_js_test.append(R2_adj(y_true=dta.test[target].values, y_pred=reversed_test[:, -1], dim1=len(list(output_spec[0].keys()))))

        report_measures_ = {'iteration': ['{0} | {1} | noNEWS | enabledRFE'.format(maskeds_coded[j], models_names[i])] * 4,
                            'fold': ['train', 'train', 'test', 'test'],
                            'measure': ['MAE', 'R2_adj', 'MAE', 'R2_adj'],
                            'value': [maes_train[-1], r2_js_train[-1], maes_test[-1], r2_js_test[-1]]}

        report_values_ = {'iteration': ['{0} | {1} | noNEWS | enabledRFE'.format(maskeds_coded[j], models_names[i])] * full_len,
                          'fold': ['train'] * train_len + ['test'] * test_len,
                          'tix': list(range(train_len)) + list(range(test_len)),
                          'y_true': list(ys_true_train[-1]) + list(ys_true_test[-1]),
                          'y_hat': list(ys_hat_train[-1]) + list(ys_hat_test[-1])}

        report_measures = report_measures.append(pandas.DataFrame(data=report_measures_), ignore_index=True)
        report_values = report_values.append(pandas.DataFrame(data=report_values_), ignore_index=True)

# pyplot.plot(range(train_len), ys_true_train[0], 'navy', range(train_len), ys_hat_train[0], 'orange')
# pyplot.plot(range(test_len), ys_true_test[0], 'navy', range(test_len), ys_hat_test[0], 'orange')

report_code = 'simpleA_ex-1'

date_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

report_measures['code'] = report_code
report_measures['date'] = date_now
report_values['code'] = report_code
report_values['date'] = date_now

# reported_n_muted(report_measures=report_measures, report_values=report_values)

print("Process Finished:\t {0}".format(datetime.datetime.utcnow().replace(tzinfo=pytz.utc)))

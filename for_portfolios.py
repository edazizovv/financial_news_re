#
import pytz
import numpy
import pandas
import datetime
from matplotlib import pyplot

#
from m_utils.transform import lag_it
from mpydge.wrap.new_data import DataHandler, MaskHandler
from mpydge.chaotic.the_new_pipe import SimplePipe
from new_insane import Insane, Neakt
from new_insane import DTR
from new_insane import MAE, R2_adj
from reporter import reported_n_muted
from pman_utils.portfolio_skeleton import PortfolioSleketon
from pman_utils.utils import portfolio_decision_dynamics_simple
from risk_utils.risk_machine import RiskMachine

#
d = './data/simplicon/data_1_LAST.csv'

dataset = pandas.read_csv(d)

dataset = dataset.set_index(['time'])
dataset = dataset.sort_index(ascending=True)

dataset = lag_it(dataset, n_lags=1, exactly=False)
dataset = dataset.dropna()

dogtag = dataset.copy()
drops = [x for x in dogtag.columns.values if 'EURUSDX_Volume' in x or 'JPYX_Volume' in x]
dogtag = dogtag.drop(columns=drops)
cols_to_drop = [x for x in dogtag.columns.values if (('LAG0' in x and 'Close' not in x) or ('Adj' in x))]
open_columns = [x for x in dogtag.columns.values if ('LAG0' in x and 'Open' in x)]
opens = dogtag[open_columns]
dogtag = dogtag.drop(columns=cols_to_drop)

ys_true_train = []
ys_true_test = []
ys_hat_train = []
ys_hat_test = []

maes_train, r2_js_train = [], []
maes_test, r2_js_test = [], []

items = [Neakt, DTR]

report_measures = pandas.DataFrame(columns=['iteration', 'fold', 'measure', 'value'])
report_values = pandas.DataFrame(columns=['iteration', 'fold', 'tix', 'y_true', 'y_hat'])

targets = [x for x in dogtag.columns.values if 'LAG0' in x]

for j in range(len(targets)):

    target = targets[j]

    qualitative = []
    quantitative = [x for x in dogtag.columns.values if 'LAG0' not in x]

    X_names = [qualitative + quantitative + [target], qualitative + quantitative]
    Y_names = [target, target]
    outputs_0 = qualitative + quantitative + [target]
    output_spec = [{x: 'float64' for x in outputs_0}, {target: 'float64'}]

    g_mask = [x in quantitative for x in X_names[0]]  # !
    target_mask = [x == target for x in X_names[0]]

    train_mask = [x < 1500 for x in range(dogtag.shape[0])]
    test_mask = [x >= 1500 for x in range(dogtag.shape[0])]
    sample_mask = MaskHandler(train=train_mask, validation=None, test=test_mask)
    dta = DataHandler(data_frame=dogtag, qualitative=qualitative, quantitative=quantitative, sample_mask=sample_mask)
    dta.sample()

    masked = {Insane(my_name='Nothing'): g_mask, Insane(my_name='Nothing'): target_mask}
    masked_coded = 'Nothing'

    items_args = [{'masked': masked, 'coded': masked_coded}, {'rfe_cv': False}]
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
    r2_js_train.append(R2_adj(y_true=dta.train[target].values, y_pred=reversed_train[:, -1],
                              dim1=len(list(output_spec[0].keys()))))
    maes_test.append(MAE(y_true=dta.test[target].values, y_pred=reversed_test[:, -1]))
    r2_js_test.append(R2_adj(y_true=dta.test[target].values, y_pred=reversed_test[:, -1],
                             dim1=len(list(output_spec[0].keys()))))

    report_measures_ = {'iteration': ['{0}'.format(target)] * 4,
                        'fold': ['train', 'train', 'test', 'test'],
                        'measure': ['MAE', 'R2_adj', 'MAE', 'R2_adj'],
                        'value': [maes_train[-1], r2_js_train[-1], maes_test[-1], r2_js_test[-1]]}

    report_values_ = {'iteration': ['{0}'.format(target)] * 1787,
                      'fold': ['train'] * 1500 + ['test'] * 287,
                      # 'tix': list(range(1500)) + list(range(287)),                  # overwritten, see below
                      'tix': dataset.index.values,
                      'y_true': list(ys_true_train[-1]) + list(ys_true_test[-1]),
                      'y_hat': list(ys_hat_train[-1]) + list(ys_hat_test[-1])}

    report_measures = report_measures.append(pandas.DataFrame(data=report_measures_), ignore_index=True)
    report_values = report_values.append(pandas.DataFrame(data=report_values_), ignore_index=True)

# reversion

report_values = report_values[report_values['fold'] == 'test']
report_values = report_values.drop(columns=['fold'])

opens = opens[numpy.isin(opens.index.values, report_values['tix'].values)]

real_values, predicted_values = report_values.drop(columns=['y_hat']), report_values.drop(columns=['y_true'])

# real_values = real_values.pivot(index=['fold', 'tix'], columns=['iteration'], values=[x for x in real_values.columns.values if x not in ['iteration', 'fold', 'tix']])
real_values = real_values.pivot(index=['tix'], columns=['iteration'],
                                values=[x for x in real_values.columns.values
                                        if x not in ['iteration', 'tix']])
# predicted_values = predicted_values.pivot(index=['fold', 'tix'], columns=['iteration'], values=[x for x in predicted_values.columns.values if x not in ['iteration', 'fold', 'tix']])
predicted_values = predicted_values.pivot(index=['tix'], columns=['iteration'],
                                          values=[x for x in predicted_values.columns.values
                                                  if x not in ['iteration', 'tix']])

real_values.columns = real_values.columns.get_level_values(level='iteration')
predicted_values.columns = predicted_values.columns.get_level_values(level='iteration')

real_values.columns = [x[:x.index('_')] for x in real_values.columns.values]
predicted_values.columns = [x[:x.index('_')] for x in predicted_values.columns.values]
opens.columns = [x[:x.index('_')] for x in opens.columns.values]

# portfolio stuff

ptf = PortfolioSleketon(real_values, predicted_values)

ptf.compute()

dec, tp, sl = ptf.get_all()

res = portfolio_decision_dynamics_simple(k0=1000, opened=opens, closed=real_values, decision=dec)

# risk utils stuff

tt = res.index.values
b = './data/simplicon/data_1_GSPC_bench_LAST.csv'
bench = pandas.read_csv(b)
# benchmark = benchmark.set_index(['time'])
# tt = benchmark['time'].values
benchmark = bench['Close'].values
benchmark = (benchmark / benchmark[0]) * 1000
portfolio = res.values

# pyplot.plot(tt, res, 'navy', tt, benchmark, 'black')
# pyplot.show()

risk_machine = RiskMachine()
risk_machine.add_benchs([benchmark])
risk_machine.add_portfolios([portfolio])
risk_machine.summary()

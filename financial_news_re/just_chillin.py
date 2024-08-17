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

from sell_stone import MutualInfoRazor

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
"""
g_mask = [x in quantitative for x in X_names[0]]  # !
target_mask = [x == target for x in X_names[0]]

# they are similar both for target and factors, remove redundant, pls
maskeds = [{Insane(my_name='Nothing'): g_mask, Insane(my_name='Nothing'): target_mask},
           {Insane(my_name='TanhLnPct'): g_mask, Insane(my_name='TanhLnPct'): target_mask},
           {Insane(my_name='LnPct'): g_mask, Insane(my_name='LnPct'): target_mask},
           {Insane(my_name='Whiten'): g_mask, Insane(my_name='Whiten'): target_mask},
           {Insane(my_name='TanhWhiten'): g_mask, Insane(my_name='TanhWhiten'): target_mask}]

maskeds_coded = ['Nothing', 'TanhLnPct', 'LnPct', 'Whiten', 'TanhWhiten']


j = 0
nk_args = {'masked': maskeds[j], 'coded': maskeds_coded[j]}
nk = Neakt(**nk_args)

nk.fit(X=dogtag.values, Y=None)
pk = nk.predict(X=dogtag.values)
"""

print('doin the stuff')

"""
pre_kwargs = {'percentile': 50}
mir = MutualInfoRazor(**pre_kwargs)
mir.fit(X=dogtag[qualitative + quantitative].values, Y=dogtag[target].values)
"""

from sklearn.feature_selection import mutual_info_regression
mired = mutual_info_regression(X=dogtag[qualitative + quantitative].values,
                               y=dogtag[target].values,
                               discrete_features='auto',
                               n_neighbors=3,
                               copy=True)

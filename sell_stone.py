#
from boruta import BorutaPy
from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_regression

#


#
class MutualInfoRazor:

    def __init__(self, percentile=50):
        self.percentile = percentile
        self.transformer = GenericUnivariateSelect(score_func=mutual_info_regression,
                                                   mode='percentile', param=self.percentile)

    @property
    def support_(self):
        return self.transformer.get_support()

    def fit(self, X, Y):
        self.transformer.fit(X, Y)

    def predict(self, X):
        return self.transformer.transform(X=X)


class BorutaRazor:

    def __init__(self, model, model_kwargs):

        self.model = model(**model_kwargs)
        self.boruta = BorutaPy(self.model)

    @property
    def support_(self):
        return self.boruta.support_

    def fit(self, X, Y):

        self.boruta.fit(X=X, y=Y)

    def predict(self, X):

        return self.boruta.transform(X=X)

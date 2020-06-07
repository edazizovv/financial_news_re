#
import numpy


#
def recursive_batch(model, model_params, X, Y, censor, X_test=None, Y_test=None, X_=None):
    names = X_ is not None
    test = X_test is not None and Y_test is not None
    if names:
        XX_ = X_.copy()
    if test:
        XX_test, YY_test = X_test.copy(), Y_test.copy()
    done = False
    XX, YY = X.copy(), Y.copy()
    mask = numpy.array([True] * XX.shape[1])
    while not done:
        XX, YY = XX[mask], YY
        if names:
            XX_ = XX_[mask]
        if test:
            XX_test, YY_test = XX_test[mask], YY_test
        mask = censor(model, XX, YY)
        done = mask.sum() == 0
    if not names and not test:
        return XX, YY
    if not names and test:
        return XX, YY, XX_test, YY_test
    if names and not test:
        return XX, YY, XX_
    if names and test:
        return XX, YY, XX_test, YY_test, XX_

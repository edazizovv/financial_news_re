#


# sampling function
def ts_sampler(T, X, Y, n_folds, test_rate, kind):
    if kind == 'nofolds':
        thresh = int((1 - test_rate) * X.shape[0])
        T_train, X_train, Y_train, T_test, X_test, Y_test = T[:thresh, :], X[:thresh, :], Y[:thresh, :], T[thresh:,
                                                                                                         :], X[thresh:,
                                                                                                             :], Y[
                                                                                                                 thresh:,
                                                                                                                 :]
    if kind == 'folded':
        thresh = int((1 - test_rate) * X.shape[0])
        T_train, X_train, Y_train, T_test, X_test, Y_test = T[:thresh, :], X[:thresh, :], Y[:thresh, :], T[thresh:,
                                                                                                         :], X[thresh:,
                                                                                                             :], Y[
                                                                                                                 thresh:,
                                                                                                                 :]
        fold_length = thresh / n_folds
        fold_bounds = [(int(j * fold_length), int((j + 1) * fold_length)) for j in range(n_folds)]
        T_train = [T_train[fold_bounds[j][0]:T_train[fold_bounds[j][1]], :] for j in range(n_folds)]
        X_train = [X_train[fold_bounds[j][0]:X_train[fold_bounds[j][1]], :] for j in range(n_folds)]
        Y_train = [Y_train[fold_bounds[j][0]:Y_train[fold_bounds[j][1]], :] for j in range(n_folds)]
    return T_train, X_train, Y_train, T_test, X_test, Y_test

# Only USE, nothing else

import time


def echo_estimator(current_time, x_time_series):
    # now no estimations provided
    return 0, 0, 0, 0, 0, 0


# it will be too slow, so we are skipping this stage: all embeddings will be calculated at the collected_aggregator
def collectie_estimator_onebyone(news_attributes, current_time, x_time_series):
    # now returns only USE embeddings

    from news_embedder.overhelm import embedding_pool
    from news_embedder.configuration import Config

    data = news_attributes.text

    config = Config()
    config.model = {}

    run_time = time.time()
    n_char = data['Text'].apply(lambda x: len(x)).sum()
    print('Size of the data being treated is:\n\tN of texts = {}\n\tTotal N of characters = {}'.format(data.shape[0],
                                                                                                       n_char))
    result_data = embedding_pool(data, ['use'], config)

    run_time = time.time() - run_time
    print('Total run time = {0:.2f} seconds'.format(run_time))

    return result_data, None, None


def collectie_estimator_skipped(news_attributes, current_time, x_time_series):
    return None, None, None


def collected_aggregator_allin(collecties, current_time, x_time_series, smoothing_factor):
    # let us just use simple inverted exponential weighting (why not?)

    # the main idea: transfer all data frames of use embeddings to numpy.ndarrays
    # swap it's axes
    # multiply by some weights among axis 1
    # sum over axis 1
    # and then use exponential smoothing over history to get next value

    from news_embedder.overhelm import embedding_pool
    from news_embedder.configuration import Config

    data = [collectie.text for collectie in collecties]
    data = pandas.DataFrame(data=data, columns=['Text'])

    config = Config()
    config.model = {}

    run_time = time.time()
    n_char = data['Text'].apply(lambda x: len(x)).sum()
    print('Size of the data being treated is:\n\tN of texts = {}\n\tTotal N of characters = {}'.format(data.shape[0],
                                                                                                       n_char))
    result_data = embedding_pool(data, ['use'], config)

    run_time = time.time() - run_time
    print('Total run time = {0:.2f} seconds'.format(run_time))

    array = result_data.values

    # so we use simply weight them equally (resulting in taking mean value)
    weights = numpy.array([1 / array.shape[0]] * array.shape[0])

    result = (numpy.swapaxes(array, 0, 1) * weights).sum(axis=1).reshape(-1, 1)

    return result

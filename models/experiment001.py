
import time
import numpy
import pandas


def stator(batches, alpha):

    vectors = (batches[j].local_vector for j in range(len(batches)))
    vectors = numpy.concatenate(vectors, axis=1)

    ss = []
    # this should be rewritten to more efficiant code!
    for j in range(vectors.shape[1]):
        if j == 0:
            s = vectors[:, j]
        else:
            s = alpha * vectors[:, j] + (1- alpha) * s
        ss.append(s)
    ss = numpy.concatenate(ss, axis=1)
    ss = numpy.swapaxes(ss, 0, 1)

    return ss


def localer(news_list):

    from news_embedder.overhelm import embedding_pool
    from news_embedder.configuration import Config

    data = [piece_of_news for piece_of_news in news_list]
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

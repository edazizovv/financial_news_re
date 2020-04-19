

# a class for time series echo
class Echo:
    def __init__(self, estimator, entities):

        self.estimator = estimator

        # the current scale is obviously arbitrary; it should be set smartly
        # may be use spectrogram?

        # immediate reaction
        self.nano = {entity: 0 for entity in entities}

        # 15-min reaction
        self.micro = {entity: 0 for entity in entities}

        # intraday reaction
        self.intra = {entity: 0 for entity in entities}

        # daily - weekly reaction
        self.med = {entity: 0 for entity in entities}

        # from several weeks to months
        self.hue = {entity: 0 for entity in entities}

        # annual (and longer) reaction
        self.macro = {entity: 0 for entity in entities}

    def estimate(self, current_time, x_time_series):

        self.nano, self.micro, self.intra, self.med, self.hue, self.macro = self.estimator(current_time, x_time_series)


class NewsAttributes:
    def __init__(self, time_stamp, title, text, source, rank):
        self.time_stamp = time_stamp
        self.title = title
        self.text = text
        self.source = source
        self.rank = rank


# a class containing news embeddings and echos
class Collectie:
    def __init__(self, estimator, time_stamp, title, text, source, rank):
        self.estimator = estimator

        self.news_attributes = NewsAttributes(time_stamp, title, text, source, rank)

        self.embeddings = []
        self.entities = {}  # here echos could be used
        self.sentiments = []

    def estimate(self, current_time, x_time_series):

        self.embeddings, self.entities, self.sentiments = self.estimator(self.news_attributes, current_time, x_time_series)


class Collected:
    def __init__(self, aggregator):
        self.collecties = []
        self.aggregator = aggregator # specifies how a state vector should be computed
        self.state_vector = None  # how to init it?

    def add(self, collectie, current_time, x_time_series):
        collectie.estimate(current_time, x_time_series)
        self.collecties.append(collectie)

    def aggregate(self, current_time, x_time_series, **kwargs):
        # here go some computations of the state vector
        self.state_vector = self.aggregator(self.collecties, current_time, x_time_series, **kwargs)


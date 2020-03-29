

class NewsBatches:
    def __init__(self, stator, stator_args, localer):
        self.batches = None
        self.stator = stator
        self.stator_args = stator_args
        self.localer = localer
        self.state_vectors = None

    def feed(self, batches):
        self.batches = batches
        
    def state(self):
        for j in range(len(self.batches)):
            self.batches[j] = self.batches[j].locale(self.localer)
        self.state_vectors = self.stator(self.batches, **self.stator_args)


class NewsBatch:
    def __init__(self, news_list):
        self.news_list = news_list
        self.local_vector = None

    def locale(self, localer):
        self.local_vector = localer(self.news_list)


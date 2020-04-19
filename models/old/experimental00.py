

class BaseModel:
    def __init__(self, state_vector_estimator, state_vector_estimator_params, batcher, sequential, sequential_params):
        # part 1. state vector estimators
        self.state_vector_estimator = state_vector_estimator
        self.state_vector_estimator_params = state_vector_estimator_params
        self.batcher = batcher
        # part 2. data augmention with the state vector
        self.data_augmentor = None
        # part 3. main pipe
        self.sequential = sequential
        self.sequential_params = sequential_params

    def fit(self, X, news, Y):
        
        # news

        batches = [self.batcher(news_list) for news_list in news]
        s_estimator = self.state_vector_estimator(**self.state_vector_estimator_params)
        s_estimator.feed(batches)
        s_estimator.state()
        
        # data augmention (now just simple concatenation)
        
        Z = X.copy()
        Z = numpy.concatenate((Z, s_estimator.state_vectors), axis=1)
        
        # main pipe
        self.sequential = self.sequential(**self.sequential_params)
        self.sequential.fit(Z, Y)
        

    # given X predict Y
    def predict(self, X, news):
                
        # news

        batches = [self.batcher(news_list) for news_list in news[j]]
        s_estimator = self.state_vector_estimator(**self.state_vector_estimator_params)
        s_estimator.feed(batches)
        s_estimator.state()
        
        # data augmention (now just simple concatenation)
        
        Z = X.copy()
        Z = numpy.concatenate((Z, s_estimator.state_vectors), axis=1)
        
        # main pipe
        self.sequential = self.sequential()
        self.sequential.fit(Z, Y)

    # predict the next T steps consequently
    def forecast(self, T):
        ...

    def test(self, X, Y, plot=False, summary=False):
        ...


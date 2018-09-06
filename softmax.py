class SoftmaxClassifier:
    def __init__(self, n_classes=3, verbose=False, tol=None, n_iters=None, alpha=5e-5, percent_change=0.00002):
        np = __import__('numpy')
        self.n_classes = n_classes           #number of classes for classification
        self.verbose = verbose               #if the user wants an update on the tasks being completed and loss function values
        self.tol = tol                          #if the user wants to stop when a particular tolerance value for the loss function is reached
        self.n_iters = n_iters                  #if the user wants to stop after a particular number of iterations
        self.alpha = alpha
        self.percent_change = percent_change
        self.insanity_check()

    def insanity_check(self):
        if self.n_classes<3 or self.n_classes % 1 != 0:
            raise ValueError('n_classes can take only natural numbers >3')          #num classes must be natural number and greater than 2
        elif self.tol is not None:
            if self.tol<0:
                raise ValueError('tolerance must be positive')                          #tolerance is positive
        elif self.n_iters is not None:
            if self.n_iters<0:
                raise ValueError('n_iters should be a positive number')                 #number of iterations should be positive

    def fit(self, X, y):                #training happens weights W is being calculated here
        np = __import__('numpy')
        X = np.append(np.array([[1]]*X.shape[0]), X, axis=1)
        self.W = np.random.rand(X.shape[1], self.n_classes)
        if self.n_iters is not  None:
            for i in range(self.n_iters):
                y_hat = X.dot(self.W)
                y_hat = self.softmax(y_hat)
                #print(y_hat)
                deriv = self.batch_grad(y_hat, y, X)
                self.W = self.W - deriv
                if self.verbose:
                    if i%50 == 0:
                        print('{} updates completed!'.format(i+1))
        elif self.tol is not None:
            y_hat = np.zeros((self.m, self.n_classes))
            while self.loss(y_hat, y)> self.tol:
                y_hat = X.dot(self.W)
                y_hat = self.softmax(y_hat)
                deriv = self.batch_grad(y_hat, y, X)
                self.W = self.W - deriv
                if self.verbose:
                    if i%50 == 0:
                        print('{} updates completed!'.format(i+1))
        else:
            collections = __import__('collections')
            self.loss_log = collections.deque([10]*100)
            pct_change = (self.loss_log[0] - self.loss_log[99])*100/self.loss_log[0]
            iters = 1
            while True:
                for i in range(100):
                    y_hat = X.dot(self.W)
                    y_hat = self.softmax(y_hat)
                    deriv = self.batch_grad(y_hat, y, X)
                    self.W = self.W - self.alpha*deriv
                    if self.verbose:
                        if iters%500 == 0:
                            print('{} updates completed!'.format(iters))
                    self.loss_log.popleft()
                    self.loss_log.append(self.loss(y_hat, y))
                    iters = iters + 1
                pct_change = (self.loss_log[0] - self.loss_log[99])*100/self.loss_log[0]
                if pct_change<self.percent_change: #and self.loss_log[99]<100:
                    break
                print('pct_change loss[0] loss[99]',pct_change, self.loss_log[0], self.loss_log[99])
        
    def batch_grad(self, y_hat, y, X):        #batch gradient for entire data set
        np = __import__('numpy')
        grad = X.T.dot(y_hat-y)
        grad = grad/y.shape[0]
        return grad
    def softmax(self, y_hat):                   #the softmax sunction
        np = __import__('numpy')
        y_hat = np.exp(y_hat)
        for i in range(y_hat.shape[0]):
            y_hat[i, :] = y_hat[i, :]/np.sum(y_hat[i, :])
        #print('y_hat', y_hat)
        return y_hat

    def loss(self, y_hat, y):           #the softmax cross entropy loss for given y_hat is calculated
        np = __import__('numpy')
        l = -y.dot(np.log(y_hat).T)
        l = np.sum(l)/y.shape[0]
        return l

    def predict(self, X, called_from_score=False):               #predicts class for an x values
        np = __import__('numpy')
        if not called_from_score:
            X = np.append(np.array([[1]]*X.shape[0]), X, axis=1)
        y_hat = X.dot(self.W)
        y_hat = self.softmax(y_hat)
        #print(y_hat)
        labels = np.argmax(y_hat, axis=1)
        y_hat = np.zeros(y_hat.shape)
        for i in range(y_hat.shape[0]):
            y_hat[i, labels[i]] = 1
        #print(y_hat)
        return y_hat

    def score(self, X, y):                     #gives the accuracy score for a given X  and y
        np = __import__('numpy')
        X = np.append(np.array([[1]]*X.shape[0]), X, axis=1)
        y_hat = self.predict(X, called_from_score=True)
        #print(np.argmax(y_hat, axis=1))
        #print(np.argmax(y, axis=1))
        acc = (np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))
        acc = np.sum(acc)/y.shape[0]
        return acc
        
    
    

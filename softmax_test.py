from softmax import SoftmaxClassifier
import numpy as np
X = np.array([[1,0,0],[0,1,0],[0,0,1]]*50)
y = X
X = X + np.random.rand(X.shape[0], X.shape[1])
clf = SoftmaxClassifier(verbose=True)
clf.fit(X,y)

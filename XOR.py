# python perceptron_xor.py

# import the necessary packages
from pyimagesearch.nn.perceptron import Perceptron
import numpy as np

# construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# define our perceptron and train it
print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

# now that our perceptron is trained we can evaluate it
print("[INFO] testing perceptron...")

# now that our network is trained, loop over the data points
for (x, target) in zip(X, y):
	# make a prediction on the data point and display the result
	# to our console
	pred = p.predict(x)
	print("[INFO] data={}, ground-truth={}, pred={}".format(
		x, target[0], pred))
	

# Conclusion:
#The XOR problem demonstrates a fundamental limitation of single-layer perceptrons: they cannot learn functions
#that are not linearly separable. Since XOR requires separating the data with a non-linear decision boundary,
#a single linear classifier will always fail. This limitation motivated the development 
#of multi-layer neural networks, where hidden layers allow the model to combine multiple linear boundaries 
#into a non-linear one. As a result, XOR became a key example showing why deeper architectures are essential
#in modern neural networks.
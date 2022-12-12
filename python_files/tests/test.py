from matplotlib import pyplot
from keras.datasets import cifar10

(trainX, trainy), (testX, testy) = cifar10.load_data()

print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

for i in range(9):

	pyplot.subplot(331 + i)

	pyplot.imshow(trainX[i])

pyplot.show()
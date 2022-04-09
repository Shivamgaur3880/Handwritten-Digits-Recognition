import pandas as pd
import numpy as np
import matplotlib as plt

# mnist = fetch_openml('mnist_784')
mnist = pd.read_csv("mnist_784.csv")
print(mnist)

x , y = mnist.loc[:,'pixel1':'pixel784'],mnist["class"]

x.shape
y.shape

# matplotlib to show image
import matplotlib.pyplot as plt

some_digit = x.iloc[4997]
some_digit

some_digit_image = some_digit.values.reshape(28,28)

plt.imshow(some_digit_image,cmap=plt.cm.binary,interpolation="nearest")
plt.show()


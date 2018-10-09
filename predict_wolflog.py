import random
from keras import models
import numpy as np

model = models.load_model('analogy_wolflog_20181008.h5')
onehot_x = np.load('onehot_x.npy')
random_i = int(random.random()*len(onehot_x))
print(onehot_x[random_i])
predict_onehot = onehot_x[random_i]
print(model.predict(predict_onehot[np.newaxis, :, :, np.newaxis]))

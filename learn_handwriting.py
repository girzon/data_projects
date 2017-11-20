from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

import pandas as pd
import numpy as np


#load training data
image_input = pd.read_csv('./train_data/imagedata.csv')
label_output = pd.read_csv('./train_data/labeldata.csv')

#load test to predict data
test_image_input = pd.read_csv('./test_data/imagedata.csv')
test_label_output = pd.read_csv('./test_data/labeldata.csv')

#print(label_output.head(20))
#print(image_input.head(20))


mlp = MLPClassifier(hidden_layer_sizes=(100,50,25),verbose=10,activation='relu',learning_rate= 'adaptive',max_iter=20)

mlp.fit(image_input,label_output)
print(mlp)

print(mlp.coefs_[0])
##do few predictions
predictions = mlp.predict(test_image_input)


#results
print(classification_report(test_label_output,predictions))

heading_names = (",".join(['O'+str(s) for s in range(0, 10)]))
np.savetxt("./test_data/prediction.csv",predictions, fmt=['%1.3f']*10, delimiter=",",header = heading_names,comments='')

#vis
#[(l.get_weights(), l.get_biases()) for l in nn.mlp.layers]
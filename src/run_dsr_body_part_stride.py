from sklearn import preprocessing

from dsr import DeepSymbolicRegressor
import matplotlib.pyplot as plt
import numpy as np
from stride_segmentation import StridesFromHws

# load data of a patient
subject_id = 'N001_AS_B_2020_01_04_09_55'
print(f'Subject : {subject_id}')
loader = StridesFromHws()
# data = [data_dx, data_sn][strides][coordinates, time][instances][body_part][x,y]
data = loader.getStridesFromHws(subject_id)

# get coordinates of nose (change 5th dimension to switch body part)
x = []
y = []
for i in range(0, len(data[0][0][0])):
    x.append(data[0][0][0][i][0][0])  # , data[0][0][1][0](tempo)
    y.append(data[0][0][0][i][0][1])

# format data for regressor
x = np.array(x)
y = np.array(y)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

scaler = preprocessing.StandardScaler().fit(x)
x = scaler.transform(x)
scaler = preprocessing.StandardScaler().fit(y)
y = scaler.transform(y)
print('x:\n', x)
print('y:\n', y)
# plot coordinates
plt.plot(x, y)
plt.show()

is_axis = False
for c in range(0, len(y)):
    print('entra')
    if y[c][0] == float(0):
        print('is 0')
        is_axis = True
    else:
        print('is not 0')
        is_axis = False
        break


if not is_axis:
    # Create the model
    model = DeepSymbolicRegressor("deep_symbolic_regressor/dsr/config.json")

    # Fit the model
    model.fit(x, y.reshape(1, -1))  # reformat y shape due to dsr different format

    # View the best expression
    print(model.program_.pretty())

    # get model y coordinates and score
    y_model = model.predict(x)
    score = model.score(x, y)  # due to use by dsr, y format for score must follow scikit-learn 1D data format
else:
    y_model = y
    score = None

# plot coordinates
plt.plot(x, y_model)
plt.text(-1, 1.25, f'score : {score}', fontsize=15)
plt.show()

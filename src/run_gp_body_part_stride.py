import numpy as np
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
from sklearn import preprocessing
import matplotlib.pyplot as plt

from stride_segmentation import StridesFromHws

# load data of a patient
subject_id = 'N001_AS_B_2020_01_04_09_55'
print(f'Subject : {subject_id}')
loader = StridesFromHws()
data = loader.getStridesFromHws(subject_id)

# get coordinates of nose
x = []
y = []
for i in range(0, len(data[0][0][0])):
    x.append(data[0][0][0][i][5][0])  # , data[0][0][1][0](tempo)
    y.append(data[0][0][0][i][5][1])

# format data for regressor
x = np.array(x)
y = np.array(y)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

x = preprocessing.StandardScaler().fit_transform(x)
y = preprocessing.StandardScaler().fit_transform(y)

# plot coordinates
plt.plot(x, y)
plt.show()

# make protected exp function in order to avoid overflow error
def _protected_exponent(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 100, np.exp(x1), 0.)


exp = make_function(function=_protected_exponent, name='exp', arity=1)

# run gp
est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, tournament_size=20,
                           stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.05, random_state=0,
                           function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min',
                                         'sin', 'cos', 'tan', exp])
est_gp.fit(x, y)

# get gp best function
gp_best_function = est_gp.program
print(gp_best_function)

y_model = est_gp.predict(x)
score = est_gp.score(x, y)
print('score : ', score)

# plot coordinates
plt.plot(x, y_model)
plt.text(-1, 1.25, f'score : {score}', fontsize=15)

plt.show()

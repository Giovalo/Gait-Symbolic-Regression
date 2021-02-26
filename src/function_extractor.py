import os

import numpy as np
import pandas as pd
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from dsr import DeepSymbolicRegressor
from stride_segmentation import StridesFromHws
from sklearn import preprocessing


def extract_strides_functions(subject_id):

    print(f'Subject : {subject_id}')

    ## LOAD DATA
    # Strides
    loader = StridesFromHws()
    data = loader.getStridesFromHws(subject_id)
    # body part names in the same order of data
    body_part_names = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow',
                      'LWrist', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle']
    # result dataframe
    subject_df = pd.DataFrame(
        columns=['side_view', 'stride_number', 'body_part', 'dsr_best_function', 'dsr_pre_visit_traversal',
                 'dsr_y', 'dsr_score', 'gp_best_function', 'gp_y', 'gp_score'])

    # for each side
    for side in range(0, len(data)):
        print(f'\t- Number of strides on %s view: {len(data[side])}' % ('right' if side == 0 else 'left'))

        # for each stride
        for i in range(0, len(data[side])):

            # for each body part
            for j in range(0, len(data[side][i][0][0])):
                print(f'\t\t- Stride {i} body part {body_part_names[j]}')

                # get coordinates
                x = []
                y = []
                for k in range(0, len(data[side][i][0])):
                    x.append(data[side][i][0][k][j][0])  # , data[0][i][1][k](tempo)
                    y.append(data[side][i][0][k][j][1])

                # format data for regressor
                x = np.array(x)
                y = np.array(y)
                x = x.reshape(-1, 1)  # need this for 1d data
                y = y.reshape(-1, 1)

                scaler = preprocessing.StandardScaler().fit(x)
                x = scaler.transform(x)
                scaler = preprocessing.StandardScaler().fit(y)
                y = scaler.transform(y)

                # Check if coordinates are all on x axis.
                # It's need to avoid regression errors
                is_axis = False
                for c in range(0, len(y)):
                    if y[c][0] == float(0):
                        is_axis = True
                    else:
                        is_axis = False
                        break

                if not is_axis:

                    # run dsr
                    print('\t\t\t- Executing deep symbolic regression...')
                    dsr = DeepSymbolicRegressor("deep_symbolic_regressor/dsr/config.json")
                    dsr.fit(x, y.reshape(1, -1))  # reformat y shape due to dsr different format
                    # get dsr best expression
                    dsr_best_funct = dsr.program_.pretty()
                    dsr_taversal = dsr.program_.traversal
                    # get dsr y coordinates and score
                    y_dsr = dsr.predict(x)
                    score_dsr = dsr.score(x, y)  # due to use by dsr, y format for score must follow skl 1D data format
                    print('\t\t\t\t- Best expression retrieved :', dsr_best_funct)
                    print('\t\t\t\t- Score :', score_dsr)

                    # create gplearn protected exp function in order to avoid overflow error
                    def _protected_exponent(x1):
                        with np.errstate(over='ignore'):
                            return np.where(np.abs(x1) < 100, np.exp(x1), 0.)

                    exp = make_function(function=_protected_exponent, name='exp', arity=1)

                    # run gp
                    print('\t\t\t- Executing genetic programming regression...')
                    est_gp = SymbolicRegressor(population_size=5000,
                                               generations=20, tournament_size=20,
                                               stopping_criteria=0.01,
                                               p_crossover=0.7, p_subtree_mutation=0.1,
                                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                                               max_samples=0.9, verbose=0,
                                               parsimony_coefficient=0.05, random_state=0,
                                               function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs',
                                                             'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan', exp])
                    est_gp.fit(x, y)
                    # get gp best function
                    gp_best_function = est_gp.program
                    # get gp y coordinates and score
                    y_gp = est_gp.predict(x)
                    score_gp = est_gp.score(x, y)
                    print('\t\t\t\t- Best expression retrieved :', gp_best_function)
                    print('\t\t\t\t- Score :', score_gp)

                else:
                    dsr_best_funct = 'is x axis'
                    dsr_taversal = None
                    y_dsr = y
                    score_dsr = 0
                    gp_best_function = 'is x axis'
                    y_gp = y
                    score_gp = 0

                # add results data to subject df
                subject_df = subject_df.append({
                    'side_view': side,
                    'stride_number': i,
                    'body_part': body_part_names[j],
                    'dsr_best_function': str(dsr_best_funct),
                    'dsr_pre_visit_traversal': dsr_taversal,
                    'dsr_y': str(y_dsr),
                    'dsr_score': score_dsr,
                    'gp_best_function': str(gp_best_function),
                    'gp_y': str(y_gp),
                    'gp_score': score_gp},
                    ignore_index=True)

                # plot coordinates
                fig, ax = plt.subplots()
                fig.suptitle(subject_id, fontsize=12, fontweight='bold')

                ax.set_title(body_part_names[j])
                ax.set_xlabel('x')
                ax.set_ylabel('y')

                ax.plot(x, y, color='blue')
                ax.plot(x, y_gp, color='green')
                ax.plot(x, y_dsr, color='red')

                # add legend
                blue_line = mlines.Line2D([], [], color='blue', label='Coordinates')
                green_line = mlines.Line2D([], [], color='green', label=f'GPLearn Exp\nScore: {score_gp:.4f}')
                red_line = mlines.Line2D([], [], color='red', label=f'DSR Exp\nScore: {score_dsr:.4f}')
                plt.legend(handles=[red_line, green_line, blue_line])

                # save plot
                save_plt_path = 'data/regressions/plot/' + subject_id
                if not os.path.exists(save_plt_path):
                    os.mkdir(save_plt_path)
                save_plt_path = save_plt_path + '/stride'
                if side is 0:
                    save_plt_path += '_dx'
                else:
                    save_plt_path += '_sx'
                save_plt_path += f'_{i}'  # stride number
                save_plt_path += f'_{body_part_names[j]}'  # boby part number # TODO : try to insert body part name
                plt.savefig(save_plt_path + '.jpeg')
                # plt.show()

    # save subject function dataframe info
    print(f'\t- Saving subject {subject_id} symbolic regression results')
    subject_df.to_csv('data/regressions/' + subject_id + '.csv', index=False)


if __name__ == '__main__':
    import time

    start_time = time.time()
    extract_strides_functions('N001_AS_B_2020_01_04_09_55')
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))

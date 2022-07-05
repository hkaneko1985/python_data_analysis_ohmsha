# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""
# サンプルプログラムで使われる関数群

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import metrics


## 目的変数の実測値と推定値との間で、散布図を描いたり、r2, RMSE, MAE を計算したりする関数
# def performance_check_in_regression(y, estimated_y):
#    plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
#    plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
#    plt.scatter(y, estimated_y.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
#    y_max = max(y.max(), estimated_y.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
#    y_min = min(y.min(), estimated_y.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
#    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
#             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
#    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
#    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
#    plt.xlabel('actual y')  # x 軸の名前
#    plt.ylabel('estimated y')  # y 軸の名前
#    plt.show()  # 以上の設定で描画
#
#    r2 = metrics.r2_score(y, estimated_y)  # r2
#    rmse = metrics.mean_squared_error(y, estimated_y) ** 0.5  # RMSE
#    mae = metrics.mean_absolute_error(y, estimated_y)  # MAE
#    return (r2, rmse, mae)


def k3n_error(x_1, x_2, k):
    """
    k-nearest neighbor normalized error (k3n-error)

    When X1 is data of X-variables and X2 is data of Z-variables
    (low-dimensional data), this is k3n error in visualization (k3n-Z-error).
    When X1 is Z-variables (low-dimensional data) and X2 is data of data of
    X-variables, this is k3n error in reconstruction (k3n-X-error).

    k3n-error = k3n-Z-error + k3n-X-error

    Parameters
    ----------
    x_1: numpy.array or pandas.DataFrame
    x_2: numpy.array or pandas.DataFrame
    k: int
        The numbers of neighbor

    Returns
    -------
    k3n_error : float
        k3n-Z-error or k3n-X-error
    """
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)

    x_1_distance = cdist(x_1, x_1)
    x_1_sorted_indexes = np.argsort(x_1_distance, axis=1)
    x_2_distance = cdist(x_2, x_2)

    for i in range(x_2.shape[0]):
        _replace_zero_with_the_smallest_positive_values(x_2_distance[i, :])

    identity_matrix = np.eye(len(x_1_distance), dtype=bool)
    knn_distance_in_x_1 = np.sort(x_2_distance[:, x_1_sorted_indexes[:, 1:k + 1]][identity_matrix])
    knn_distance_in_x_2 = np.sort(x_2_distance)[:, 1:k + 1]

    sum_k3n_error = (
            (knn_distance_in_x_1 - knn_distance_in_x_2) / knn_distance_in_x_2
    ).sum()
    return sum_k3n_error / x_1.shape[0] / k


def _replace_zero_with_the_smallest_positive_values(arr):
    """
    Replace zeros in array with the smallest positive values.

    Parameters
    ----------
    arr: numpy.array
    """
    arr[arr == 0] = np.min(arr[arr != 0])


def plot_and_selection_of_hyperparameter(hyperparameter_values, metrics_values, x_label, y_label):
    # ハイパーパラメータ (成分数、k-NN の k など) の値ごとの統計量 (CV 後のr2, 正解率など) をプロット
    plt.rcParams['font.size'] = 18
    plt.scatter(hyperparameter_values, metrics_values, c='blue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    # 統計量 (CV 後のr2, 正解率など) が最大のときのハイパーパラメータ (成分数、k-NN の k など) の値を選択
    return hyperparameter_values[metrics_values.index(max(metrics_values))]


def estimation_and_performance_check_in_regression_train_and_test(model, x_train, y_train, x_test, y_test):
    # トレーニングデータの推定
    estimated_y_train = model.predict(x_train) * y_train.std() + y_train.mean()  # y を推定し、スケールをもとに戻します
    estimated_y_train = pd.DataFrame(estimated_y_train, index=x_train.index,
                                     columns=['estimated y'])  # Pandas の DataFrame 型に変換。行の名前・列の名前も設定
    
    # トレーニングデータの実測値 vs. 推定値のプロット
    plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
    plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
    plt.scatter(y_train, estimated_y_train.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
    y_max = max(y_train.max(), estimated_y_train.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
    y_min = min(y_train.min(), estimated_y_train.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
    plt.xlabel('actual y')  # x 軸の名前
    plt.ylabel('estimated y')  # y 軸の名前
    plt.show()  # 以上の設定で描画

    # トレーニングデータのr2, RMSE, MAE
    print('r^2 for training data :', metrics.r2_score(y_train, estimated_y_train))
    print('RMSE for training data :', metrics.mean_squared_error(y_train, estimated_y_train) ** 0.5)
    print('MAE for training data :', metrics.mean_absolute_error(y_train, estimated_y_train))

    # トレーニングデータの結果の保存
    y_train_for_save = pd.DataFrame(y_train)  # Series のため列名は別途変更
    y_train_for_save.columns = ['actual y']
    y_error_train = y_train_for_save.iloc[:, 0] - estimated_y_train.iloc[:, 0]
    y_error_train = pd.DataFrame(y_error_train)  # Series のため列名は別途変更
    y_error_train.columns = ['error of y(actual y - estimated y)']
    results_train = pd.concat([estimated_y_train, y_train_for_save, y_error_train], axis=1)
    results_train.to_csv('estimated_y_train.csv')  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

    # テストデータの推定
    estimated_y_test = model.predict(x_test) * y_train.std() + y_train.mean()  # y を推定し、スケールをもとに戻します
    estimated_y_test = pd.DataFrame(estimated_y_test, index=x_test.index,
                                    columns=['estimated y'])  # Pandas の DataFrame 型に変換。行の名前・列の名前も設定
   
    # テストデータの実測値 vs. 推定値のプロット
    plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
    plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
    plt.scatter(y_test, estimated_y_test.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
    y_max = max(y_test.max(), estimated_y_test.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
    y_min = min(y_test.min(), estimated_y_test.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
    plt.xlabel('actual y')  # x 軸の名前
    plt.ylabel('estimated y')  # y 軸の名前
    plt.show()  # 以上の設定で描画

    # テストデータのr2, RMSE, MAE
    print('r^2 for test data :', metrics.r2_score(y_test, estimated_y_test))
    print('RMSE for test data :', metrics.mean_squared_error(y_test, estimated_y_test) ** 0.5)
    print('MAE for test data :', metrics.mean_absolute_error(y_test, estimated_y_test))

    # テストデータの結果の保存
    y_test_for_save = pd.DataFrame(y_test)  # Series のため列名は別途変更
    y_test_for_save.columns = ['actual y']
    y_error_test = y_test_for_save.iloc[:, 0] - estimated_y_test.iloc[:, 0]
    y_error_test = pd.DataFrame(y_error_test)  # Series のため列名は別途変更
    y_error_test.columns = ['error of y (actual y - estimated y)']
    results_test = pd.concat([estimated_y_test, y_test_for_save, y_error_test], axis=1)
    results_test.to_csv('estimated_y_test.csv')  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

def estimation_and_performance_check_in_classification_train_and_test(model, x_train, y_train, x_test, y_test):
    class_types = list(set(y_train))  # クラスの種類。これで混同行列における縦と横のクラスの順番を定めます
    class_types.sort(reverse=True)  # 並び替え

    # トレーニングデータのクラスの推定
    estimated_y_train = pd.DataFrame(model.predict(x_train), index=x_train.index, columns=[
        'estimated class'])  # トレーニングデータのクラスを推定し、Pandas の DataFrame 型に変換。行の名前・列の名前も設定

    # トレーニングデータの混同行列
    confusion_matrix_train = pd.DataFrame(
        metrics.confusion_matrix(y_train, estimated_y_train, labels=class_types), index=class_types,
        columns=class_types)  # 混同行列を作成し、Pandas の DataFrame 型に変換。行の名前・列の名前を定めたクラスの名前として設定
    confusion_matrix_train.to_csv('confusion_matrix_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
    print(confusion_matrix_train)  # 混同行列の表示
    print('Accuracy for training data :', metrics.accuracy_score(y_train, estimated_y_train), '\n')  # 正解率の表示

    # トレーニングデータの結果の保存
    y_train_for_save = pd.DataFrame(y_train)  # Series のため列名は別途変更
    y_train_for_save.columns = ['actual class']
    y_error_train = y_train_for_save.iloc[:, 0] == estimated_y_train.iloc[:, 0]
    y_error_train = pd.DataFrame(y_error_train)  # Series のため列名は別途変更
    y_error_train.columns = ['TRUE if estimated class is correct']
    results_train = pd.concat([estimated_y_train, y_train_for_save, y_error_train], axis=1)
    results_train.to_csv('estimated_y_train.csv')  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

    # テストデータのクラスの推定
    estimated_y_test = pd.DataFrame(model.predict(x_test), index=x_test.index,
                                    columns=['estimated class'])  # テストデータのクラスを推定し、Pandas の DataFrame 型に変換。行の名前・列の名前も設定

    # テストデータの混同行列
    confusion_matrix_test = pd.DataFrame(
        metrics.confusion_matrix(y_test, estimated_y_test, labels=class_types), index=class_types,
        columns=class_types)  # 混同行列を作成し、Pandas の DataFrame 型に変換。行の名前・列の名前を定めたクラスの名前として設定
    confusion_matrix_test.to_csv('confusion_matrix_test.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
    print(confusion_matrix_test)  # 混同行列の表示
    print('Accuracy for test data :', metrics.accuracy_score(y_test, estimated_y_test))  # 正解率の表示

    # テストデータの結果の保存
    y_test_for_save = pd.DataFrame(y_test)  # Series のため列名は別途変更
    y_test_for_save.columns = ['actual class']
    y_error_test = y_test_for_save.iloc[:, 0] == estimated_y_test.iloc[:, 0]
    y_error_test = pd.DataFrame(y_error_test)  # Series のため列名は別途変更
    y_error_test.columns = ['TRUE if estimated class is correct']
    results_test = pd.concat([estimated_y_test, y_test_for_save, y_error_test], axis=1)
    results_test.to_csv('estimated_y_test.csv')  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

def add_nonlinear_terms(x):
    """
    DataFrame型の x に、二乗項と交差項を追加して出力する関数

    Parameters
    ----------
    x: pandas.DataFrame

    Returns
    -------
    x: pandas.DataFrame
    
    """

    original_x = x.copy()  # 元の説明変数のデータセット
    x_square = x ** 2  # 二乗項
    # 追加
    print('\n二乗項と交差項の追加')
    for i in range(original_x.shape[1]):
        print(i + 1, '/', original_x.shape[1])
        for j in range(original_x.shape[1]):
            if i == j:  # 二乗項
                x = pd.concat(
                    [x, x_square.rename(columns={x_square.columns[i]: '{0}^2'.format(x_square.columns[i])}).iloc[:, i]],
                    axis=1)
            elif i < j:  # 交差項
                x_cross = original_x.iloc[:, i] * original_x.iloc[:, j]
                x_cross.name = '{0}*{1}'.format(original_x.columns[i], original_x.columns[j])
                x = pd.concat([x, x_cross], axis=1)
    return x

def add_time_delayed_variable(x, dynamics_max, dynamics_span):
    """
    DataFrame型もしくは array 型の x に、時間遅れ変数を追加して出力する関数

    Parameters
    ----------
    x: pandas.DataFrame or numpy.array

    Returns
    -------
    x: pandas.DataFrame or numpy.array
    
    """

    x_array = np.array(x)
    if dynamics_max:
        x_with_dynamics = np.empty((x_array.shape[0] - dynamics_max, 0 ))
        x_with_dynamics = np.append(x_with_dynamics, x_array[dynamics_max:, 0:1], axis=1)
        for x_variable_number in range(x_array.shape[1] - 1):
            x_with_dynamics = np.append(x_with_dynamics, x_array[dynamics_max:, x_variable_number+1:x_variable_number+2], axis=1)
            for time_delay_number in range(int(np.floor(dynamics_max / dynamics_span))):
                x_with_dynamics = np.append(x_with_dynamics, x_array[dynamics_max-(time_delay_number+1)*dynamics_span:-(time_delay_number+1)*dynamics_span, x_variable_number+1:x_variable_number+2], axis=1)
    else:
        x_with_dynamics = x_array.copy()
    return x_with_dynamics

def lwpls(x_train, y_train, x_test, max_component_number, lambda_in_similarity):
    """
    Locally-Weighted Partial Least Squares (LWPLS)
    
    Predict y-values of test samples using LWPLS

    Parameters
    ----------
    x_train: numpy.array or pandas.DataFrame
        autoscaled m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y_train: numpy.array or pandas.DataFrame
        autoscaled m x 1 vector of a Y-variable of training data
    x_test: numpy.array or pandas.DataFrame
        k x n matrix of X-variables of test data, which is autoscaled with training data,
        and k is the number of test samples
    max_component_number: int
        number of maximum components
    lambda_in_similarity: float
        parameter in similarity matrix

    Returns
    -------
    estimated_y_test : numpy.array
        k x 1 vector of estimated y-values of test data
    """

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = np.reshape(y_train, (len(y_train), 1))
    x_test = np.array(x_test)

    estimated_y_test = np.zeros((x_test.shape[0], max_component_number))
    distance_matrix = cdist(x_train, x_test, 'euclidean')
    for test_sample_number in range(x_test.shape[0]):
        query_x_test = x_test[test_sample_number, :]
        query_x_test = np.reshape(query_x_test, (1, len(query_x_test)))
        distance = distance_matrix[:, test_sample_number]
        similarity = np.diag(np.exp(-distance / distance.std(ddof=1) / lambda_in_similarity))
        #        similarity_matrix = np.diag(similarity)

        y_w = y_train.T.dot(np.diag(similarity)) / similarity.sum()
        x_w = np.reshape(x_train.T.dot(np.diag(similarity)) / similarity.sum(), (1, x_train.shape[1]))
        centered_y = y_train - y_w
        centered_x = x_train - np.ones((x_train.shape[0], 1)).dot(x_w)
        centered_query_x_test = query_x_test - x_w
        estimated_y_test[test_sample_number, :] += y_w
        for component_number in range(max_component_number):
            w_a = np.reshape(centered_x.T.dot(similarity).dot(centered_y) / np.linalg.norm(
                centered_x.T.dot(similarity).dot(centered_y)), (x_train.shape[1], 1))
            t_a = np.reshape(centered_x.dot(w_a), (x_train.shape[0], 1))
            p_a = np.reshape(centered_x.T.dot(similarity).dot(t_a) / t_a.T.dot(similarity).dot(t_a),
                             (x_train.shape[1], 1))
            q_a = centered_y.T.dot(similarity).dot(t_a) / t_a.T.dot(similarity).dot(t_a)
            t_q_a = centered_query_x_test.dot(w_a)
            estimated_y_test[test_sample_number, component_number:] = estimated_y_test[test_sample_number,
                                                                                       component_number:] + t_q_a * q_a
            if component_number != max_component_number:
                centered_x = centered_x - t_a.dot(p_a.T)
                centered_y = centered_y - t_a * q_a
                centered_query_x_test = centered_query_x_test - t_q_a.dot(p_a.T)

    return estimated_y_test

def gamma_optimization_with_variance(x, gammas):
    """
    DataFrame型もしくは array 型の x において、カーネル関数におけるグラム行列の分散を最大化することによって
    γ を最適化する関数

    Parameters
    ----------
    x: pandas.DataFrame or numpy.array
    gammas: list

    Returns
    -------
    optimized gamma : scalar
    
    """
    print('カーネル関数において、グラム行列の分散を最大化することによる γ の最適化')
    variance_of_gram_matrix = list()
    for index, ocsvm_gamma in enumerate(gammas):
        print(index + 1, '/', len(gammas))
        gram_matrix = np.exp(-ocsvm_gamma * cdist(x, x, metric='sqeuclidean'))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    return gammas[variance_of_gram_matrix.index(max(variance_of_gram_matrix))]

def structure_generation_based_on_r_group_random(file_name_of_main_fragments, file_name_of_sub_fragments, number_of_structures):
    from rdkit import Chem

    bond_list = [Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                 Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.QUADRUPLE, Chem.rdchem.BondType.QUINTUPLE,
                 Chem.rdchem.BondType.HEXTUPLE, Chem.rdchem.BondType.ONEANDAHALF, Chem.rdchem.BondType.TWOANDAHALF,
                 Chem.rdchem.BondType.THREEANDAHALF, Chem.rdchem.BondType.FOURANDAHALF, Chem.rdchem.BondType.FIVEANDAHALF,
                 Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.IONIC, Chem.rdchem.BondType.HYDROGEN,
                 Chem.rdchem.BondType.THREECENTER, Chem.rdchem.BondType.DATIVEONE, Chem.rdchem.BondType.DATIVE,
                 Chem.rdchem.BondType.DATIVEL, Chem.rdchem.BondType.DATIVER, Chem.rdchem.BondType.OTHER,
                 Chem.rdchem.BondType.ZERO]
    
    main_molecules = [molecule for molecule in Chem.SmilesMolSupplier(file_name_of_main_fragments, delimiter='\t', titleLine=False) if molecule is not None]
    fragment_molecules = [molecule for molecule in Chem.SmilesMolSupplier(file_name_of_sub_fragments, delimiter='\t', titleLine=False) if molecule is not None]
    
    print('主骨格のフラグメントの数 :', len(main_molecules))
    print('側鎖のフラグメントの数 :', len(fragment_molecules))

    generated_structures = []
    for generated_structure_number in range(number_of_structures):
        selected_main_molecule_number = np.floor(np.random.rand(1) * len(main_molecules)).astype(int)[0]
        main_molecule = main_molecules[selected_main_molecule_number]
        # make adjacency matrix and get atoms for main molecule
        main_adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(main_molecule)
        for bond in main_molecule.GetBonds():
            main_adjacency_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_list.index(bond.GetBondType())
            main_adjacency_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond_list.index(bond.GetBondType())
        main_atoms = []
        for atom in main_molecule.GetAtoms():
            main_atoms.append(atom.GetSymbol())
        
        r_index_in_main_molecule_old = [index for index, atom in enumerate(main_atoms) if atom == '*']
        for index, r_index in enumerate(r_index_in_main_molecule_old):
            modified_index = r_index - index
            atom = main_atoms.pop(modified_index)
            main_atoms.append(atom)
            tmp = main_adjacency_matrix[:, modified_index:modified_index + 1].copy()
            main_adjacency_matrix = np.delete(main_adjacency_matrix, modified_index, 1)
            main_adjacency_matrix = np.c_[main_adjacency_matrix, tmp]
            tmp = main_adjacency_matrix[modified_index:modified_index + 1, :].copy()
            main_adjacency_matrix = np.delete(main_adjacency_matrix, modified_index, 0)
            main_adjacency_matrix = np.r_[main_adjacency_matrix, tmp]
        r_index_in_main_molecule_new = [index for index, atom in enumerate(main_atoms) if atom == '*']
        
        r_bonded_atom_index_in_main_molecule = []
        for number in r_index_in_main_molecule_new:
            r_bonded_atom_index_in_main_molecule.append(np.where(main_adjacency_matrix[number, :] != 0)[0][0])
        r_bond_number_in_main_molecule = main_adjacency_matrix[
            r_index_in_main_molecule_new, r_bonded_atom_index_in_main_molecule]
        
        main_adjacency_matrix = np.delete(main_adjacency_matrix, r_index_in_main_molecule_new, 0)
        main_adjacency_matrix = np.delete(main_adjacency_matrix, r_index_in_main_molecule_new, 1)
        
        for i in range(len(r_index_in_main_molecule_new)):
            main_atoms.remove('*')
        main_size = main_adjacency_matrix.shape[0]
        
        selected_fragment_numbers = np.floor(np.random.rand(len(r_index_in_main_molecule_old)) * len(fragment_molecules)).astype(int)
          
        generated_molecule_atoms = main_atoms[:]
        generated_adjacency_matrix = main_adjacency_matrix.copy()
        for r_number_in_molecule in range(len(r_index_in_main_molecule_new)):
            fragment_molecule = fragment_molecules[selected_fragment_numbers[r_number_in_molecule]]
            fragment_adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(fragment_molecule)
            for bond in fragment_molecule.GetBonds():
                fragment_adjacency_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_list.index(
                    bond.GetBondType())
                fragment_adjacency_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond_list.index(
                    bond.GetBondType())
            fragment_atoms = []
            for atom in fragment_molecule.GetAtoms():
                fragment_atoms.append(atom.GetSymbol())
    
            # integrate adjacency matrix
            r_index_in_fragment_molecule = fragment_atoms.index('*')
    
            r_bonded_atom_index_in_fragment_molecule = \
                np.where(fragment_adjacency_matrix[r_index_in_fragment_molecule, :] != 0)[0][0]
            if r_bonded_atom_index_in_fragment_molecule > r_index_in_fragment_molecule:
                r_bonded_atom_index_in_fragment_molecule -= 1
    
            fragment_atoms.remove('*')
            fragment_adjacency_matrix = np.delete(fragment_adjacency_matrix, r_index_in_fragment_molecule, 0)
            fragment_adjacency_matrix = np.delete(fragment_adjacency_matrix, r_index_in_fragment_molecule, 1)
        
            main_size = generated_adjacency_matrix.shape[0]
            generated_adjacency_matrix = np.c_[generated_adjacency_matrix, np.zeros(
                [generated_adjacency_matrix.shape[0], fragment_adjacency_matrix.shape[0]], dtype='int32')]
            generated_adjacency_matrix = np.r_[generated_adjacency_matrix, np.zeros(
                [fragment_adjacency_matrix.shape[0], generated_adjacency_matrix.shape[1]], dtype='int32')]
    
            generated_adjacency_matrix[r_bonded_atom_index_in_main_molecule[
                                           r_number_in_molecule], r_bonded_atom_index_in_fragment_molecule + main_size] = \
                r_bond_number_in_main_molecule[r_number_in_molecule]
            generated_adjacency_matrix[
                r_bonded_atom_index_in_fragment_molecule + main_size, r_bonded_atom_index_in_main_molecule[
                    r_number_in_molecule]] = r_bond_number_in_main_molecule[r_number_in_molecule]
            generated_adjacency_matrix[main_size:, main_size:] = fragment_adjacency_matrix
    
            # integrate atoms
            generated_molecule_atoms += fragment_atoms
    
        # generate structures 
        generated_molecule = Chem.RWMol()
        atom_index = []
        for atom_number in range(len(generated_molecule_atoms)):
            atom = Chem.Atom(generated_molecule_atoms[atom_number])
            molecular_index = generated_molecule.AddAtom(atom)
            atom_index.append(molecular_index)
        for index_x, row_vector in enumerate(generated_adjacency_matrix):
            for index_y, bond in enumerate(row_vector):
                if index_y <= index_x:
                    continue
                if bond == 0:
                    continue
                else:
                    generated_molecule.AddBond(atom_index[index_x], atom_index[index_y], bond_list[bond])
    
        generated_molecule = generated_molecule.GetMol()
        generated_structures.append(Chem.MolToSmiles(generated_molecule))
        if (generated_structure_number + 1) % 1000 == 0 or (generated_structure_number + 1) == number_of_structures:
            print(generated_structure_number + 1, '/', number_of_structures)
    
    return generated_structures
    
    

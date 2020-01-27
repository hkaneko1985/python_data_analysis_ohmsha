# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import sample_functions
from sklearn import metrics
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV

# 下の y_name を、'boiling_point', 'logS', 'melting_point', 'pIC50', 'pIGC50' のいずれかにしてください。
# descriptors_with_[y_name].csv というファイルを dataset として読み込み計算します。
# さらに、y_name を別の名前に変えて、ご自身で別途 sample_program_6_8_0_csv.py もしくは
# sample_program_6_8_0_sdf.py で descriptors_with_[y_name].csv というファイルを、
# 他のファイルと同様の形式で準備すれば、同じように計算することができます。

y_name = 'logS'
# 'boiling_point' : 沸点のデータセットの場合
# 'logS' : 水溶解度のデータセットの場合
# 'melting_point' : 融点のデータセットの場合
# 'pIC50' : 薬理活性のデータセットの場合
# 'pIGC50' : 環境毒性のデータセットの場合

structures_name = 'file'  # 'file' or 'brics' or 'r_group' or 'descriptors'
# 'file' : 予測用の化学構造を読み込み、y の値を推定します。
#          file_name_for_prediction を、予測用の化学構造のファイル名にしてください。
#          csv ファイルもしくは sdf ファイルです。 サンプルとして、molecules_for_prediction.csv,
#          molecules_for_prediction.sdf, molecules_estimated_pIC50_positive.csv があります。
#
# 'brics' : BRICS アルゴリズムで生成された化学構造の y の値を推定します。
#           化学構造生成の元となる構造のデータセットのファイル名を、file_name_of_seed_structures で指定してください。
#
# 'r_group' : R-group の化学構造をランダムに生成して y の値を推定します。
#             予測用の化学構造を発生するため、主骨格のフラグメントのファイル名を file_name_of_main_fragments で、
#             側鎖のフラグメントのファイル名を file_name_of_sub_fragments で指定してください。どちらも SMILES にしてください。
#
# 'descriptors' : 予測用の化学構造の記述子データセットを読み込み、y の値を推定します。
#                 予測用のデータセットの csv ファイル名を、file_name_of_descriptors_for_prediction で指定してください。
#                 このファイルは、事前に sample_program_6_8_6_descriotprs_for_prediction.py で計算する必要があります。

file_name_for_prediction = 'molecules_for_prediction.csv'  # 'file', 'descriptors' 予測用のデータセットのファイル名
#file_name_for_prediction = 'molecules_estimated_pIC50_positive.csv'  # 'file', 'descriptors' 予測用のデータセットのファイル名
file_name_of_seed_structures = 'molecules_with_{0}.csv'.format(y_name)  # 'brics' 構造生成のための元になる化学構造のファイル名。csv ファイルか sdf ファイルです。
#file_name_of_seed_structures = 'molecules_for_prediction.csv'  # 'brics' 構造生成のための元になる化学構造のファイル名。csv ファイルか sdf ファイルです。
file_name_of_main_fragments = 'sample_main_fragments.smi'  # 'r_group' 主骨格のフラグメントがあるファイル名。サンプルとして、'sample_main_fragments.smi' があります。
#file_name_of_main_fragments = 'sample_main_fragments_for_pIC50.smi'  # 'r_group' 主骨格のフラグメントがあるファイル名。サンプルとして、'sample_main_fragments.smi' があります。
file_name_of_sub_fragments = 'sample_sub_fragments.smi'  # 'r_group' 側鎖のフラグメントがあるファイル名。サンプルとして、'sample_main_fragments.smi' があります
number_of_generated_structures = 10000  # 'brics', 'r_group' 生成する化学構造の数
file_name_of_descriptors_for_prediction = 'descriptors_of_molecules_for_prediction.csv'  # 'descriptors' 記述子データセットのファイル名

method_name = 'svr'  # 'pls' or 'svr'
ad_method_name = 'ensemble'  # 'ensemble' or 'ocsvm' or 'no'(ADなし)
add_nonlinear_terms_flag = False  # True (二乗項・交差項を追加) or False (追加しない)
number_of_bins = 50  # y の推定値におけるヒストグラムのビンの数

fold_number = 5  # N-fold CV の N
max_number_of_principal_components = 30  # 使用する主成分の最大数
svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # C の候補
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # ε の候補
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補
ocsvm_nu = 0.003  # OCSVM における ν。トレーニングデータにおけるサンプル数に対する、サポートベクターの数の下限の割合
ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補
number_of_submodels = 50  # サブモデルの数
rate_of_selected_x_variables = 0.8  # 各サブデータセットで選択される説明変数の数の割合。0 より大きく 1 未満

if structures_name != 'file' and structures_name != 'brics' and structures_name != 'r_group' and structures_name != 'descriptors':
    sys.exit('\'{0}\' という予測用の化学構造(生成)はありません。structures_name を見直してください。'.format(structures_name))
if method_name != 'pls' and method_name != 'svr':
    sys.exit('\'{0}\' という回帰分析手法はありません。method_name を見直してください。'.format(method_name))
if ad_method_name != 'ensemble' and ad_method_name != 'ocsvm' and ad_method_name != 'no':
    sys.exit('\'{0}\' というAD設定手法はありません。ad_method_name を見直してください。'.format(ad_method_name))
    
dataset = pd.read_csv('descriptors_with_{0}.csv'.format(y_name), index_col=0)  # 物性・活性と記述子のデータセットの読み込み
y = dataset.iloc[:, 0]
original_x = dataset.iloc[:, 1:]
original_x = original_x.replace(np.inf, np.nan).fillna(np.nan)  # inf を NaN に置き換え
nan_variable_flags = original_x.isnull().any()  # NaN を含む変数
original_x = original_x.drop(original_x.columns[nan_variable_flags], axis=1)  # NaN を含む変数を削除
# 標準偏差が 0 の説明変数を削除
std_0_variable_flags = original_x.std() == 0
original_x = original_x.drop(original_x.columns[std_0_variable_flags], axis=1)

if add_nonlinear_terms_flag:
    x = pd.read_csv('x_{0}.csv'.format(y_name), index_col=0)  # 物性・活性と記述子のデータセットの読み込み
#    x = sample_functions.add_nonlinear_terms(x)  # 説明変数の二乗項や交差項を追加
    # 標準偏差が 0 の説明変数を削除
    std_0_nonlinear_variable_flags = x.std() == 0
    x = x.drop(x.columns[std_0_nonlinear_variable_flags], axis=1)
else:
    x = original_x.copy()

# オートスケーリング
autoscaled_original_x = (original_x - original_x.mean()) / original_x.std()
autoscaled_x = (x - x.mean()) / x.std()
autoscaled_y = (y - y.mean()) / y.std()

if ad_method_name == 'ocsvm':
    # グラム行列の分散を最大化することによる γ の最適化
    optimal_ocsvm_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x, ocsvm_gammas)

if method_name == 'pls':
    # CV による成分数の最適化
    components = []  # 空の list の変数を作成して、成分数をこの変数に追加していきます同じく成分数をこの変数に追加
    r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
    for component in range(1, min(np.linalg.matrix_rank(autoscaled_x), max_number_of_principal_components) + 1):
        # PLS
        model = PLSRegression(n_components=component)  # PLS モデルの宣言
        estimated_y_in_cv = pd.DataFrame(cross_val_predict(model, autoscaled_x, autoscaled_y,
                                                           cv=fold_number))  # クロスバリデーション推定値の計算し、DataFrame型に変換
        estimated_y_in_cv = estimated_y_in_cv * y.std() + y.mean()  # スケールをもとに戻す
        r2_in_cv = metrics.r2_score(y, estimated_y_in_cv)  # r2 を計算
        print(component, r2_in_cv)  # 成分数と r2 を表示
        r2_in_cv_all.append(r2_in_cv)  # r2 を追加
        components.append(component)  # 成分数を追加

    # 成分数ごとの CV 後の r2 をプロットし、CV 後のr2が最大のときを最適成分数に
    optimal_component_number = sample_functions.plot_and_selection_of_hyperparameter(components, r2_in_cv_all,
                                                                                     'number of components',
                                                                                     'cross-validated r2')
    print('\nCV で最適化された成分数 :', optimal_component_number)
    # PLS
    model = PLSRegression(n_components=optimal_component_number)  # モデルの宣言
elif method_name == 'svr':
    # グラム行列の分散を最大化することによる γ の最適化
    if ad_method_name == 'ocsvm':
        optimal_svr_gamma = optimal_ocsvm_gamma.copy()
    else:
        optimal_svr_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x, svr_gammas)
    # CV による ε の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                               cv=fold_number, verbose=2)
    model_in_cv.fit(autoscaled_x, autoscaled_y)
    optimal_svr_epsilon = model_in_cv.best_params_['epsilon']
    # CV による C の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                               {'C': svr_cs}, cv=fold_number, verbose=2)
    model_in_cv.fit(autoscaled_x, autoscaled_y)
    optimal_svr_c = model_in_cv.best_params_['C']
    # CV による γ の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                               {'gamma': svr_gammas}, cv=fold_number, verbose=2)
    model_in_cv.fit(autoscaled_x, autoscaled_y)
    optimal_svr_gamma = model_in_cv.best_params_['gamma']
    # 最適化された C, ε, γ
    print('C : {0}\nε : {1}\nGamma : {2}'.format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma))
    # SVR
    model = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)  # モデルの宣言

model.fit(autoscaled_x, autoscaled_y)  # モデルの構築
if method_name == 'pls':
    # 標準回帰係数
    standard_regression_coefficients = pd.DataFrame(model.coef_, index=x.columns,
                                                    columns=['standard_regression_coefficients'])
    standard_regression_coefficients.to_csv(
        'pls_standard_regression_coefficients.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
    
# 予測用の化学構造
if structures_name == 'file':  # 予測用の化学構造の読み込み
    if file_name_for_prediction[-4:] == '.csv':
        dataset_prediction = pd.read_csv(file_name_for_prediction, index_col=0)  # SMILES 付きデータセットの読み込み
        smiles_prediction = dataset_prediction.iloc[:, 0]  # 分子の SMILES
        print('分子の数 :', len(smiles_prediction))
        molecules_prediction = [Chem.MolFromSmiles(smiles_i) for smiles_i in smiles_prediction]
    elif file_name_for_prediction[-4:] == '.sdf':
        molecules_prediction = Chem.SDMolSupplier(file_name_for_prediction)  # sdf ファイルの読み込み
        print('分子の数 :', len(molecules_prediction))
elif structures_name == 'brics':  # BRICS による化学構造生成
    if file_name_of_seed_structures[-4:] == '.csv':  # SMILES で分子の読み込み
        dataset_seed = pd.read_csv(file_name_of_seed_structures, index_col=0)
        smiles_seed = dataset_seed.iloc[:, 0]  # 分子の SMILES
        molecules = [Chem.MolFromSmiles(smiles_i) for smiles_i in smiles_seed]
    elif file_name_of_seed_structures[-4:] == '.sdf':  # SDF ファイルで分子の読み込み
        molecules = Chem.SDMolSupplier(file_name_of_seed_structures)
    # フラグメントへの変換
    print('読み込んだ分子の数 :', len(molecules))
    print('フラグメントへの分解')
    fragments = set()
    for molecule in molecules:
        fragment = BRICS.BRICSDecompose(molecule, minFragmentSize=1)
        fragments.update(fragment)
    print('生成されたフラグメントの数 :', len(fragments))
    generated_structures = BRICS.BRICSBuild([Chem.MolFromSmiles(fragment) for fragment in fragments])
    # リスト型の変数に分子を格納
    molecules_prediction = []
    for index, generated_structure in enumerate(generated_structures):
        print(index + 1, '/', number_of_generated_structures)
        generated_structure.UpdatePropertyCache(True)
        AllChem.Compute2DCoords(generated_structure)
        molecules_prediction.append(generated_structure)
        if index + 1 >= number_of_generated_structures:
            break
elif structures_name == 'r_group':  # R-group の化学構造生成
    print('化学構造生成 開始')
    smiles_prediction = sample_functions.structure_generation_based_on_r_group_random(file_name_of_main_fragments,
                                                                                      file_name_of_sub_fragments,
                                                                                      number_of_generated_structures)
    molecules_prediction = [Chem.MolFromSmiles(smiles_i) for smiles_i in smiles_prediction]

if structures_name == 'descriptors':
    original_x_prediction = pd.read_csv(file_name_of_descriptors_for_prediction, index_col=0)  # 予測用の記述子のデータセットの読み込み
else:
    # 記述子の計算
    print('記述子の計算 開始')
    # 計算する記述子名の取得
    descriptor_names = []
    for descriptor_information in Descriptors.descList:
        descriptor_names.append(descriptor_information[0])
    print('計算する記述子の数 :', len(descriptor_names))
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    # 分子ごとに、リスト型の変数に計算された記述子の値や、SMILES を追加
    descriptors_of_molecules_prediction, smiles_prediction = [], []
    for index, molecule_prediction in enumerate(molecules_prediction):
        print(index + 1, '/', len(molecules_prediction))
        if molecule_prediction is not None:
            smiles_prediction.append(Chem.MolToSmiles(molecule_prediction))
            AllChem.Compute2DCoords(molecule_prediction)
            descriptors_of_molecules_prediction.append(descriptor_calculator.CalcDescriptors(molecule_prediction))
    if structures_name == 'file' and file_name_for_prediction[-4:] == '.csv':
        original_x_prediction = pd.DataFrame(descriptors_of_molecules_prediction, index=dataset_prediction.index, columns=descriptor_names)
    else:
        original_x_prediction = pd.DataFrame(descriptors_of_molecules_prediction, index=smiles_prediction, columns=descriptor_names)
#original_x_prediction = original_x_prediction.drop(['MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge'], axis=1)
original_x_prediction = original_x_prediction.replace(np.inf, np.nan).fillna(np.nan)  # inf を nan に置き換え
original_x_prediction = original_x_prediction.drop(original_x_prediction.columns[nan_variable_flags], axis=1)
original_x_prediction = original_x_prediction.drop(original_x_prediction.columns[std_0_variable_flags], axis=1)
nan_sammple_flags = original_x_prediction.isnull().any(axis=1)  # NaN を含むサンプル
original_x_prediction = original_x_prediction.drop(original_x_prediction.index[nan_sammple_flags], axis=0)  # NaN を含むサンプル
if add_nonlinear_terms_flag:
    x_prediction = sample_functions.add_nonlinear_terms(original_x_prediction)  # 説明変数の二乗項や交差項を追加
    # 標準偏差が 0 の説明変数を削除
    x_prediction = x_prediction.drop(x_prediction.columns[std_0_nonlinear_variable_flags], axis=1)
else:
    x_prediction = original_x_prediction.copy()

print('予測と AD の検討開始')
# オートスケーリング
autoscaled_original_x_prediction = (original_x_prediction - original_x.mean()) / original_x.std()
autoscaled_x_prediction = (x_prediction - x.mean()) / x.std()
# y の値の予測
estimated_y_for_prediction = pd.DataFrame(model.predict(autoscaled_x_prediction) * y.std() + y.mean(),
                                          index=x_prediction.index, columns=[dataset.columns[0]])
# AD
if ad_method_name == 'ocsvm':
    # OCSVM による AD
    print('最適化された gamma (OCSVM) :', optimal_ocsvm_gamma)
    ad_model = svm.OneClassSVM(kernel='rbf', gamma=optimal_ocsvm_gamma, nu=ocsvm_nu)  # AD モデルの宣言
    ad_model.fit(autoscaled_original_x)  # モデル構築
    # 予測用データに対して、AD の中か外かを判定
    data_density_prediction = ad_model.decision_function(autoscaled_original_x_prediction)  # データ密度 (f(x) の値)
    estimated_y_for_prediction = estimated_y_for_prediction.iloc[data_density_prediction >= 0, :].copy()
elif ad_method_name == 'ensemble':
    # アンサンブル学習による AD
    number_of_x_variables = int(np.ceil(x.shape[1] * rate_of_selected_x_variables))
    print('各サブデータセットの説明変数の数 :', number_of_x_variables)
    estimated_y_train_all = pd.DataFrame()  # 空の DataFrame 型を作成し、ここにサブモデルごとのトレーニングデータの y の推定結果を追加
    selected_x_variable_numbers = []  # 空の list 型の変数を作成し、ここに各サブデータセットの説明変数の番号を追加
    submodels = []  # 空の list 型の変数を作成し、ここに構築済みの各サブモデルを追加
    for submodel_number in range(number_of_submodels):
        print(submodel_number + 1, '/', number_of_submodels)  # 進捗状況の表示
        # 説明変数の選択
        # 0 から 1 までの間に一様に分布する乱数を説明変数の数だけ生成して、その乱数値が小さい順に説明変数を選択
        random_x_variables = np.random.rand(x.shape[1])
        selected_x_variable_numbers_tmp = random_x_variables.argsort()[:number_of_x_variables]
        selected_autoscaled_x = autoscaled_x.iloc[:, selected_x_variable_numbers_tmp]
        selected_x_variable_numbers.append(selected_x_variable_numbers_tmp)
    
        if method_name == 'pls':
            # CV による成分数の最適化
            components = []  # 空の list の変数を作成して、成分数をこの変数に追加していきます同じく成分数をこの変数に追加
            r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
            for component in range(1, min(np.linalg.matrix_rank(selected_autoscaled_x),
                                          max_number_of_principal_components) + 1):
                # PLS
                submodel_in_cv = PLSRegression(n_components=component)  # PLS モデルの宣言
                estimated_y_in_cv = pd.DataFrame(cross_val_predict(submodel_in_cv, selected_autoscaled_x, autoscaled_y,
                                                                   cv=fold_number))  # クロスバリデーション推定値の計算し、DataFrame型に変換
                estimated_y_in_cv = estimated_y_in_cv * y.std() + y.mean()  # スケールをもとに戻す
                r2_in_cv = metrics.r2_score(y, estimated_y_in_cv)  # r2 を計算
                r2_in_cv_all.append(r2_in_cv)  # r2 を追加
                components.append(component)  # 成分数を追加
            optimal_component_number = components[r2_in_cv_all.index(max(r2_in_cv_all))]
            # PLS
            submodel = PLSRegression(n_components=optimal_component_number)  # モデルの宣言
        elif method_name == 'svr':
            # ハイパーパラメータの最適化
            # CV による ε の最適化
            model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                                       cv=fold_number, iid=False)
            model_in_cv.fit(selected_autoscaled_x, autoscaled_y)
            optimal_svr_epsilon = model_in_cv.best_params_['epsilon']
            # CV による C の最適化
            model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                                       {'C': svr_cs}, cv=fold_number, iid=False)
            model_in_cv.fit(selected_autoscaled_x, autoscaled_y)
            optimal_svr_c = model_in_cv.best_params_['C']
            # CV による γ の最適化
            model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                                       {'gamma': svr_gammas}, cv=fold_number, iid=False)
            model_in_cv.fit(selected_autoscaled_x, autoscaled_y)
            optimal_svr_gamma = model_in_cv.best_params_['gamma']
            # SVR
            submodel = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon,
                               gamma=optimal_svr_gamma)  # モデルの宣言
        submodel.fit(selected_autoscaled_x, autoscaled_y)  # モデルの構築
        submodels.append(submodel)
    
    # サブデータセットの説明変数の種類やサブモデルを保存。同じ名前のファイルがあるときは上書きされるため注意
    pd.to_pickle(selected_x_variable_numbers, 'selected_x_variable_numbers.bin')
    pd.to_pickle(submodels, 'submodels.bin')
    
    # サブデータセットの説明変数の種類やサブモデルを読み込み
    # 今回は、保存した後にすぐ読み込んでいるため、あまり意味はありませんが、サブデータセットの説明変数の種類やサブモデルを
    # 保存しておくことで、後で新しいサンプルを予測したいときにモデル構築の過程を省略できます
    selected_x_variable_numbers = pd.read_pickle('selected_x_variable_numbers.bin')
    submodels = pd.read_pickle('submodels.bin')
    # y の値の標準偏差の予測
    estimated_y_for_prediction_all = pd.DataFrame()  # 空の DataFrame 型を作成し、ここにサブモデルごとのテストデータの y の推定結果を追加
    for submodel_number in range(number_of_submodels):
        # 説明変数の選択
        selected_autoscaled_x_prediction = autoscaled_x_prediction.iloc[:, selected_x_variable_numbers[submodel_number]]
        # テストデータの y の推定
        estimated_y_for_prediction_sub = pd.DataFrame(
            submodels[submodel_number].predict(selected_autoscaled_x_prediction))  # テストデータの y の値を推定し、Pandas の DataFrame 型に変換
        estimated_y_for_prediction_sub = estimated_y_for_prediction_sub * y.std() + y.mean()  # スケールをもとに戻します
        estimated_y_for_prediction_all = pd.concat([estimated_y_for_prediction_all, estimated_y_for_prediction_sub], axis=1)
    estimated_y_for_prediction = pd.DataFrame(estimated_y_for_prediction_all.median(axis=1))  # Series 型のため、行名と列名の設定は別に
    estimated_y_for_prediction.index = x_prediction.index
    estimated_y_for_prediction.columns = [dataset.columns[0]]
    std_of_estimated_y_for_prediction = pd.DataFrame(estimated_y_for_prediction_all.std(axis=1))  # Series 型のため、行名と列名の設定は別に
    std_of_estimated_y_for_prediction.index = x_prediction.index
    std_of_estimated_y_for_prediction.columns = ['std_of_estimated_y']
    estimated_y_for_prediction = pd.concat([estimated_y_for_prediction, std_of_estimated_y_for_prediction], axis=1)
elif ad_method_name == 'no':
    print('AD はありません')

# 結果の保存
estimated_y_for_prediction.to_csv('estimated_y_prediction.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# y の推定値のヒストグラム
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.hist(estimated_y_for_prediction.iloc[:, 0], bins=number_of_bins)  # ヒストグラムの作成
plt.xlabel(estimated_y_for_prediction.columns[0])  # 横軸の名前
plt.ylabel('frequency')  # 縦軸の名前
plt.show()  # 以上の設定において、グラフを描画

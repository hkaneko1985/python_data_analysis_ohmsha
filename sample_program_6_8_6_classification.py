# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import math

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import sample_functions
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# 下の y_name を 'pIC50_class', 'pIGC50_class' のいずれかにしてください。
# descriptors_with_[y_name].csv というファイルを dataset として読み込み計算します。
# さらに、y_name を別の名前に変えて、ご自身で別途 sample_program_6_8_0_csv.py もしくは
# sample_program_6_8_0_sdf.py で descriptors_with_[y_name].csv というファイルを、
# 他のファイルと同様の形式で準備すれば、同じように計算することができます。

y_name = 'pIC50_class'
# 'pIC50_class' : クラス分類用の薬理活性のデータセットの場合
# 'pIGC50_class' : クラス分類用の環境毒性のデータセットの場合

structures_name = 'r_group'  # 'file' or 'brics' or 'r_group' or 'descriptors'
# 'file' : 予測用の化学構造を読み込み、y の値を推定します。
#          file_name_for_prediction を、予測用の化学構造のファイル名にしてください。
#          csv ファイルもしくは sdf ファイルです。 サンプルとして、molecules_for_prediction.csv と
#          molecules_for_prediction.sdf がありますのでご確認ください。
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
file_name_of_seed_structures = 'molecules_with_{0}.csv'.format(y_name)  # 'brics' 構造生成のための元になる化学構造のファイル名。csv ファイルか sdf ファイルです。
#file_name_of_seed_structures = 'molecules_for_prediction.csv'  # 'brics' 構造生成のための元になる化学構造のファイル名。csv ファイルか sdf ファイルです。
file_name_of_main_fragments = 'sample_main_fragments_for_pIC50.smi'  # 'r_group' 主骨格のフラグメントがあるファイル名。サンプルとして、'sample_main_fragments.smi' があります。
file_name_of_sub_fragments = 'sample_sub_fragments.smi'  # 'r_group' 側鎖のフラグメントがあるファイル名。サンプルとして、'sample_main_fragments.smi' があります
number_of_generated_structures = 10000  # 'brics', 'r_group' 生成する化学構造の数
file_name_of_descriptors_for_prediction = 'descriptors_of_molecules_for_prediction.csv'  # 'descriptors' 記述子データセットのファイル名

method_name = 'rf'  # 'knn' or 'svm' or 'rf'
ad_method_name = 'ensemble'  # 'ensemble' or 'ocsvm' or 'no'
add_nonlinear_terms_flag = False  # True (二乗項・交差項を追加) or False (追加しない)

fold_number = 5  # N-fold CV の N
max_number_of_k = 20  # 使用する k の最大値
svm_cs = 2 ** np.arange(-5, 11, dtype=float)
svm_gammas = 2 ** np.arange(-20, 11, dtype=float)
rf_number_of_trees = 300  # RF における決定木の数
rf_x_variables_rates = np.arange(1, 11, dtype=float) / 10  # 1 つの決定木における説明変数の数の割合の候補
ocsvm_nu = 0.05  # OCSVM における ν。トレーニングデータにおけるサンプル数に対する、サポートベクターの数の下限の割合
ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補
number_of_submodels = 50  # サブモデルの数
rate_of_selected_x_variables = 0.8  # 各サブデータセットで選択される説明変数の数の割合。0 より大きく 1 未満

dataset = pd.read_csv('descriptors_with_{0}.csv'.format(y_name), index_col=0)  # 物性・活性と記述子のデータセットの読み込み
y = dataset.iloc[:, 0]
class_types = list(set(y))  # クラスの種類
class_types.sort(reverse=True)  # 並び替え
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

if ad_method_name == 'ocsvm':
    # グラム行列の分散を最大化することによる γ の最適化
    optimal_ocsvm_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x, ocsvm_gammas)

if method_name == 'knn':
    # CV による k の最適化
    accuracy_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の 正解率 をこの変数に追加していきます
    ks = []  # 同じく k の値をこの変数に追加していきます
    for k in range(1, max_number_of_k + 1):
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')  # k-NN モデルの宣言
        # クロスバリデーション推定値の計算し、DataFrame型に変換
        estimated_y_in_cv = pd.DataFrame(cross_val_predict(model, autoscaled_x, y,
                                                                           cv=fold_number))
        accuracy_in_cv = metrics.accuracy_score(y, estimated_y_in_cv)  # 正解率を計算
        print(k, accuracy_in_cv)  # k の値と r2 を表示
        accuracy_in_cv_all.append(accuracy_in_cv)  # r2 を追加
        ks.append(k)  # k の値を追加
    # k の値ごとの CV 後の正解率をプロットし、CV 後の正解率が最大のときを k の最適値に
    optimal_k = sample_functions.plot_and_selection_of_hyperparameter(ks, accuracy_in_cv_all, 'k',
                                                                      'cross-validated accuracy')
    print('\nCV で最適化された k :', optimal_k, '\n')
    # k-NN
    model = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean')  # モデルの宣言
elif method_name == 'svm':
    # グラム行列の分散を最大化することによる γ の最適化
    if ad_method_name == 'ocsvm':
        optimal_svm_gamma = optimal_ocsvm_gamma.copy()
    else:
        optimal_svm_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x, svm_gammas)
    # CV による C の最適化
    model_in_cv = GridSearchCV(svm.SVC(kernel='rbf', gamma=optimal_svm_gamma),
                               {'C': svm_cs}, cv=fold_number, iid=False, verbose=2)
    model_in_cv.fit(autoscaled_x, y)
    optimal_svm_c = model_in_cv.best_params_['C']
    # CV による γ の最適化
    model_in_cv = GridSearchCV(svm.SVC(kernel='rbf', C=optimal_svm_c),
                               {'gamma': svm_gammas}, cv=fold_number, iid=False, verbose=2)
    model_in_cv.fit(autoscaled_x, y)
    optimal_svm_gamma = model_in_cv.best_params_['gamma']
    print('CV で最適化された C :', optimal_svm_c)
    print('CV で最適化された γ:', optimal_svm_gamma)
    # SVM
    model = svm.SVC(kernel='rbf', C=optimal_svm_c, gamma=optimal_svm_gamma)  # モデルの宣言
elif method_name == 'rf':
    # OOB (Out-Of-Bugs) による説明変数の数の割合の最適化
    accuracy_oob = []
    for index, x_variables_rate in enumerate(rf_x_variables_rates):
        print(index + 1, '/', len(rf_x_variables_rates))
        model_in_validation = RandomForestClassifier(n_estimators=rf_number_of_trees, max_features=int(
            max(math.ceil(autoscaled_x.shape[1] * x_variables_rate), 1)), oob_score=True)
        model_in_validation.fit(autoscaled_x, y)
        accuracy_oob.append(model_in_validation.oob_score_)
    optimal_x_variables_rate = sample_functions.plot_and_selection_of_hyperparameter(rf_x_variables_rates,
                                                                                     accuracy_oob,
                                                                                     'rate of x-variables',
                                                                                     'accuracy for OOB')
    print('\nOOB で最適化された説明変数の数の割合 :', optimal_x_variables_rate)
    # RF
    model = RandomForestClassifier(n_estimators=rf_number_of_trees,
                                   max_features=int(
                                       max(math.ceil(autoscaled_x.shape[1] * optimal_x_variables_rate), 1)),
                                   oob_score=True)  # RF モデルの宣言

model.fit(autoscaled_x, y)  # モデルの構築
if method_name == 'rf':
    # 記述子の重要度
    x_importances = pd.DataFrame(model.feature_importances_, index=x.columns, columns=['importance'])
    x_importances.to_csv('rf_x_importances.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
    
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
        fragment = BRICS.BRICSDecompose(molecule, minFragmentSize=2)
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
estimated_y_for_prediction = pd.DataFrame(model.predict(autoscaled_x_prediction),
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

        if method_name == 'knn':
            # CV による k の最適化
            accuracy_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の 正解率 をこの変数に追加していきます
            ks = []  # 同じく k の値をこの変数に追加していきます
            for k in range(1, max_number_of_k + 1):
                model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')  # k-NN モデルの宣言
                # クロスバリデーション推定値の計算し、DataFrame型に変換
                estimated_y_in_cv = pd.DataFrame(cross_val_predict(model, selected_autoscaled_x, y,
                                                                                   cv=fold_number))
                accuracy_in_cv = metrics.accuracy_score(y, estimated_y_in_cv)  # 正解率を計算
                accuracy_in_cv_all.append(accuracy_in_cv)  # r2 を追加
                ks.append(k)  # k の値を追加
            optimal_k = ks[accuracy_in_cv_all.index(max(accuracy_in_cv_all))]
            submodel = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean')  # k-NN モデルの宣言
        elif method_name == 'svm':
            # CV による C の最適化
            model_in_cv = GridSearchCV(svm.SVC(kernel='rbf', gamma=optimal_svm_gamma),
                                       {'C': svm_cs}, cv=fold_number, iid=False)
            model_in_cv.fit(selected_autoscaled_x, y)
            optimal_svm_c = model_in_cv.best_params_['C']
            # CV による γ の最適化
            model_in_cv = GridSearchCV(svm.SVC(kernel='rbf', C=optimal_svm_c),
                                       {'gamma': svm_gammas}, cv=fold_number, iid=False)
            model_in_cv.fit(selected_autoscaled_x, y)
            optimal_svm_gamma = model_in_cv.best_params_['gamma']
            submodel = svm.SVC(kernel='rbf', C=optimal_svm_c, gamma=optimal_svm_gamma)  # SVM モデルの宣言
        elif method_name == 'rf':
            # OOB (Out-Of-Bugs) による説明変数の数の割合の最適化
            accuracy_oob = []
            for index, x_variables_rate in enumerate(rf_x_variables_rates):
                model_in_validation = RandomForestClassifier(n_estimators=rf_number_of_trees, max_features=int(
                    max(math.ceil(selected_autoscaled_x.shape[1] * x_variables_rate), 1)), oob_score=True)
                model_in_validation.fit(selected_autoscaled_x, y)
                accuracy_oob.append(model_in_validation.oob_score_)
            optimal_x_variables_rate = rf_x_variables_rates[accuracy_oob.index(max(accuracy_oob))]
            submodel = RandomForestClassifier(n_estimators=rf_number_of_trees,
                                              max_features=int(max(
                                                  math.ceil(selected_autoscaled_x.shape[1] * optimal_x_variables_rate),
                                                  1)),
                                              oob_score=True)  # RF モデルの宣言
        submodel.fit(selected_autoscaled_x, y)  # モデルの構築
        submodels.append(submodel)

    # サブデータセットの説明変数の種類やサブモデルを保存。同じ名前のファイルがあるときは上書きされるため注意
    pd.to_pickle(selected_x_variable_numbers, 'selected_x_variable_numbers.bin')
    pd.to_pickle(submodels, 'submodels.bin')

    # サブデータセットの説明変数の種類やサブモデルを読み込み
    # 今回は、保存した後にすぐ読み込んでいるため、あまり意味はありませんが、サブデータセットの説明変数の種類やサブモデルを
    # 保存しておくことで、後で新しいサンプルを予測したいときにモデル構築の過程を省略できます
    selected_x_variable_numbers = pd.read_pickle('selected_x_variable_numbers.bin')
    submodels = pd.read_pickle('submodels.bin')

    # 予測用データの y の推定
    # estimated_y_test_all = pd.DataFrame()  # 空の DataFrame 型を作成し、ここにサブモデルごとのテストデータの y の推定結果を追加
    estimated_y_test_count = np.zeros([x_prediction.shape[0], len(class_types)])  # クラスごとに、推定したサブモデルの数をカウントして値をここに格納
    for submodel_number in range(number_of_submodels):
        # 説明変数の選択
        selected_autoscaled_x_test = autoscaled_x_prediction.iloc[:, selected_x_variable_numbers[submodel_number]]

        # 予測用データの y の推定
        estimated_y_test = pd.DataFrame(
            submodels[submodel_number].predict(selected_autoscaled_x_test))  # テストデータの y の値を推定し、Pandas の DataFrame 型に変換
        #    estimated_y_test_all = pd.concat([estimated_y_test_all, estimated_y_test], axis=1)
        for sample_number in range(estimated_y_test.shape[0]):
            estimated_y_test_count[sample_number, class_types.index(estimated_y_test.iloc[sample_number, 0])] += 1
    # 予測用データにおける、クラスごとの推定したサブモデルの数
    estimated_y_test_count = pd.DataFrame(estimated_y_test_count, index=x_prediction.index, columns=class_types)
    estimated_y_test_count.to_csv('estimated_y_test_count.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

    # 予測用データにおける、クラスごとの確率
    estimated_y_test_probability = estimated_y_test_count / number_of_submodels
    estimated_y_test_probability.to_csv(
        'estimated_y_test_probability.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
#    # 多数決で推定された結果
#    estimated_y_for_prediction = pd.DataFrame(estimated_y_test_count.idxmax(axis=1), columns=[dataset.columns[0]])
elif ad_method_name == 'no':
    print('AD はありません')
    
# 結果の保存
estimated_y_for_prediction.to_csv('estimated_y_prediction.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.simplefilter(action='ignore')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('nilm/nilm/anonimized/25day_dataset.csv', sep=',')
timestamps = df.index.tolist()

df_small = df[
    ['ActivePower', 'ReactivePower', 'Voltage', 'Current', 'harmonic1_Real', 'harmonic1_Imaginary',
     'harmonic3_Real',
     'harmonic3_Imaginary', 'harmonic5_Real', 'harmonic5_Imaginary', 'wahing_machine', 'dishwasher', 'oven']]

harmonic1_mod = np.sqrt(df['harmonic1_Real'] ** 2 + df['harmonic1_Imaginary'] ** 2)
harmonic3_mod = np.sqrt(df['harmonic3_Real'] ** 2 + df['harmonic3_Imaginary'] ** 2)
harmonic5_mod = np.sqrt(df['harmonic5_Real'] ** 2 + df['harmonic5_Imaginary'] ** 2)

df_small['harmonic1_mod'] = harmonic1_mod
df_small['harmonic3_mod'] = harmonic3_mod
df_small['harmonic5_mod'] = harmonic5_mod

'''
df_small.loc[:, 'wahing_machine'] = df_small['wahing_machine'].apply(lambda x: 1 if x > 0 else 0)
df_small.loc[:, 'dishwasher'] = df_small['dishwasher'].apply(lambda x: 1 if x > 0 else 0)
df_small.loc[:, 'oven'] = df_small['oven'].apply(lambda x: 1 if x > 0 else 0)
'''

kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(df_small)
results_KNN = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])
results_RF = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall', 'Samples'])

init = time.time()
for train_index, test_index in kf.split(df_small):
    X_train, X_test = df_small.iloc[train_index], df_small.iloc[test_index]
    X_train = X_train.drop(['dishwasher', 'oven'], axis=1)
    X_test = X_test.drop(['dishwasher', 'oven'], axis=1)
    min_washing_machine = X_train['wahing_machine'].min()
    max_washing_machine = X_train['wahing_machine'].max()
    mean_washing_machine = (min_washing_machine + max_washing_machine) / 2
    # voglio dividere nuovamente in due tra min e mean
    mean_washing_machine_lower = (min_washing_machine + mean_washing_machine) / 4
    print(mean_washing_machine_lower)
    X_train.loc[:, 'wahing_machine'] = X_train['wahing_machine'].apply(lambda x: 1 if x > 0 else 0)
    X_train_upper = X_train[X_train['ActivePower'] > mean_washing_machine]
    X_train_lower_l = X_train[X_train['ActivePower'] <= mean_washing_machine_lower]
    X_train_lower_u = X_train[(X_train['ActivePower'] > mean_washing_machine_lower) & (X_train['ActivePower'] <= mean_washing_machine)]

    '''
    X_train_lower = X_train_lower.drop(
        X_train_lower[(X_train_lower['wahing_machine'] == 0)].sample(
            frac=0.60).index)
    '''

    y_train_washing_machine_upper = X_train_upper['wahing_machine']
    y_train_washing_machine_lower_l = X_train_lower_l['wahing_machine']
    y_train_washing_machine_lower_u = X_train_lower_u['wahing_machine']

    X_test.loc[:, 'wahing_machine'] = X_test['wahing_machine'].apply(lambda x: 1 if x > 0 else 0)
    X_test_upper = X_test[X_test['ActivePower'] > mean_washing_machine]
    X_test_lower_l = X_test[X_test['ActivePower'] <= mean_washing_machine_lower]
    X_test_lower_u = X_test[(X_test['ActivePower'] > mean_washing_machine_lower) & (X_test['ActivePower'] <= mean_washing_machine)]

    y_test_washing_machine_upper = X_test_upper['wahing_machine']
    y_test_washing_machine_lower_l = X_test_lower_l['wahing_machine']
    y_test_washing_machine_lower_u = X_test_lower_u['wahing_machine']

    X_train_upper = X_train_upper.drop(['wahing_machine'], axis=1)
    X_train_lower_l = X_train_lower_l.drop(['wahing_machine'], axis=1)
    X_train_lower_u = X_train_lower_u.drop(['wahing_machine'], axis=1)
    X_test_upper = X_test_upper.drop(['wahing_machine'], axis=1)
    X_test_lower_l = X_test_lower_l.drop(['wahing_machine'], axis=1)
    X_test_lower_u = X_test_lower_u.drop(['wahing_machine'], axis=1)

    scaler_upper = MinMaxScaler()
    scaler_upper.fit(X_train_upper)
    X_train_norm_upper = scaler_upper.transform(X_train_upper)
    X_test_norm_upper = scaler_upper.transform(X_test_upper)

    scaler_lower_l = MinMaxScaler()
    scaler_lower_l.fit(X_train_lower_l)
    X_train_norm_lower_l = scaler_lower_l.transform(X_train_lower_l)
    X_test_norm_lower_l = scaler_lower_l.transform(X_test_lower_l)

    scaler_lower_u = MinMaxScaler()
    scaler_lower_u.fit(X_train_lower_u)
    X_train_norm_lower_u = scaler_lower_u.transform(X_train_lower_u)
    X_test_norm_lower_u = scaler_lower_u.transform(X_test_lower_u)

    print('Random Forest Upper')
    rf_washing_machine_upper = RandomForestClassifier(n_estimators=20, random_state=0, verbose=1)
    rf_washing_machine_upper.fit(X_train_norm_upper, y_train_washing_machine_upper)
    y_pred_washing_machine_upper = rf_washing_machine_upper.predict(X_test_norm_upper)
    accuracy_wm_rf_upper = accuracy_score(y_test_washing_machine_upper, y_pred_washing_machine_upper)
    precision_wm_rf_upper = precision_score(y_test_washing_machine_upper, y_pred_washing_machine_upper)
    recall_wm_rf_upper = recall_score(y_test_washing_machine_upper, y_pred_washing_machine_upper)
    results_RF = results_RF.append({'Class': 'Washing Machine Upper', 'Accuracy': accuracy_wm_rf_upper,
                                        'Precision': precision_wm_rf_upper, 'Recall': recall_wm_rf_upper, 'Samples': len(X_test_upper)}, ignore_index=True)

    print('Random Forest Lower L')
    weights = np.ones(len(y_train_washing_machine_lower_l))
    weights[y_train_washing_machine_lower_l == 1] = 100
    rf_washing_machine_lower_l = RandomForestClassifier(n_estimators=20, random_state=0, verbose=1)
    rf_washing_machine_lower_l.fit(X_train_norm_lower_l, y_train_washing_machine_lower_l, sample_weight=weights)
    y_pred_washing_machine_lower_l = rf_washing_machine_lower_l.predict(X_test_norm_lower_l)
    accuracy_wm_rf_lower_l = accuracy_score(y_test_washing_machine_lower_l, y_pred_washing_machine_lower_l)
    precision_wm_rf_lower_l = precision_score(y_test_washing_machine_lower_l, y_pred_washing_machine_lower_l)
    recall_wm_rf_lower_l = recall_score(y_test_washing_machine_lower_l, y_pred_washing_machine_lower_l)
    results_RF = results_RF.append({'Class': 'Washing Machine Lower L', 'Accuracy': accuracy_wm_rf_lower_l,
                                        'Precision': precision_wm_rf_lower_l, 'Recall': recall_wm_rf_lower_l, 'Samples': len(X_test_norm_lower_l)}, ignore_index=True)

    print('Random Forest Lower U')
    rf_washing_machine_lower_u = RandomForestClassifier(n_estimators=20, random_state=0, verbose=1)
    rf_washing_machine_lower_u.fit(X_train_norm_lower_u, y_train_washing_machine_lower_u)
    y_pred_washing_machine_lower_u = rf_washing_machine_lower_u.predict(X_test_norm_lower_u)
    accuracy_wm_rf_lower_u = accuracy_score(y_test_washing_machine_lower_u, y_pred_washing_machine_lower_u)
    precision_wm_rf_lower_u = precision_score(y_test_washing_machine_lower_u, y_pred_washing_machine_lower_u)
    recall_wm_rf_lower_u = recall_score(y_test_washing_machine_lower_u, y_pred_washing_machine_lower_u)
    results_RF = results_RF.append({'Class': 'Washing Machine Lower U', 'Accuracy': accuracy_wm_rf_lower_u,
                                        'Precision': precision_wm_rf_lower_u, 'Recall': recall_wm_rf_lower_u, 'Samples' : len(X_test_norm_lower_u)}, ignore_index=True)

    y_test_washing_machine = np.concatenate((y_test_washing_machine_upper, y_test_washing_machine_lower_l, y_test_washing_machine_lower_u))
    y_pred_washing_machine = np.concatenate((y_pred_washing_machine_upper, y_pred_washing_machine_lower_l, y_pred_washing_machine_lower_u))

    accuracy_wm_rf = accuracy_score(y_test_washing_machine, y_pred_washing_machine)
    precision_wm_rf = precision_score(y_test_washing_machine, y_pred_washing_machine)
    recall_wm_rf = recall_score(y_test_washing_machine, y_pred_washing_machine)

    results_RF = results_RF.append({'Class': 'Washing Machine', 'Accuracy': accuracy_wm_rf,
                                        'Precision': precision_wm_rf, 'Recall': recall_wm_rf}, ignore_index=True)


end = time.time()
print("Tempo: ", (end - init) / 60, " minuti")

results_KNN['F1'] = 2 * (results_KNN['Precision'] * results_KNN['Recall']) / (
        results_KNN['Precision'] + results_KNN['Recall'])
results_RF['F1'] = 2 * (results_RF['Precision'] * results_RF['Recall']) / (
        results_RF['Precision'] + results_RF['Recall'])

results_KNN_mean = results_KNN.groupby(['Class']).mean()
results_RF_mean = results_RF.groupby(['Class']).mean()

results_KNN.to_csv('Results/Results_KNN.csv', index_label=results_KNN.index.name)
results_KNN_mean.to_csv('Results/Results_KNN_mean.csv', index_label=results_KNN_mean.index.name)
results_RF.to_csv('Results/Results_RF.csv', index_label=results_RF.index.name)
results_RF_mean.to_csv('Results/Results_RF_mean.csv', index_label=results_RF_mean.index.name)


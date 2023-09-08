import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
# from sdv.single_table import CTGANSynthesizer
# from sdv.metadata import SingleTableMetadata
# from sdmetrics.reports.single_table import QualityReport
# from sdv.sampling import Condition
# from imblearn.over_sampling import SMOTE
import warnings
import winsound

warnings.simplefilter(action='ignore', category=FutureWarning)
init = time.time()

# Carichiamo il CSV
df = pd.read_csv('nilm/anonimized/25day_dataset.csv', sep=',')

df_small = df[
    ['DateTime', 'ActivePower', 'ReactivePower', 'Voltage', 'Current', 'harmonic1_Real', 'harmonic1_Imaginary',
     'harmonic3_Real',
     'harmonic3_Imaginary', 'harmonic5_Real', 'harmonic5_Imaginary', 'wahing_machine', 'dishwasher', 'oven']]

# trasformiamo i valori delle 3 colonne target in 0 e 1
df_small.loc[:, 'wahing_machine'] = df_small['wahing_machine'].apply(lambda x: 1 if x > 0 else 0)
df_small.loc[:, 'dishwasher'] = df_small['dishwasher'].apply(lambda x: 1 if x > 0 else 0)
df_small.loc[:, 'oven'] = df_small['oven'].apply(lambda x: 1 if x > 0 else 0)

# Eliminiamo i NaN
df_small = df_small.dropna()

"""
# Essendo il dataset un po' rumoroso, proviamo a pulirlo, usando un algoritmo di smoothing, come un filtro passa-basso
df_small['ActivePower'] = df_small['ActivePower'].rolling(10).mean()
df_small['ReactivePower'] = df_small['ReactivePower'].rolling(10).mean()
df_small['Voltage'] = df_small['Voltage'].rolling(10).mean()
df_small['Current'] = df_small['Current'].rolling(10).mean()

# Grafichiamo i dati con plotly, voglio vedere se il filtro ha funzionato, facciamo solo il primo giorno però, 
# cioè i primi 86400 secondi
timestamps = df.index.tolist()
fig = go.Figure()
fig.add_trace(go.Scatter(x=timestamps[:86400], y=df_small['ActivePower'][:86400], mode='lines', name='ActivePower'))
fig.add_trace(go.Scatter(x=timestamps[:86400], y=df_small['ReactivePower'][:86400], mode='lines', name='ReactivePower'))
fig.add_trace(go.Scatter(x=timestamps[:86400], y=df_small['Voltage'][:86400], mode='lines', name='Voltage'))
fig.add_trace(go.Scatter(x=timestamps[:86400], y=df_small['Current'][:86400], mode='lines', name='Current'))
# anche la label voglio nel grafico
fig.add_trace(
    go.Scatter(x=timestamps[:86400], y=df_small['wahing_machine'][:86400], mode='lines', name='wahing_machine'))
fig.add_trace(go.Scatter(x=timestamps[:86400], y=df_small['dishwasher'][:86400], mode='lines', name='dishwasher'))
fig.add_trace(go.Scatter(x=timestamps[:86400], y=df_small['oven'][:86400], mode='lines', name='oven'))
# salviamola come html
# fig.write_html('plotly.html', auto_open=True)
"""

# creo i gruppi corrispondenti ai 25 giorni
day_groups = pd.to_datetime(df_small.DateTime).dt.day.values

# creo le 10 fold con i gruppi
kf = GroupKFold(n_splits=10)

results_KNN = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])
results_SVM = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])
results_DT = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])
results_MLP = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])
results_AdaBoost = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])


# KFold
for train_index, test_index in kf.split(df_small, groups=day_groups):  # per ogni fold
    # i dataset di training e test per la fold
    X_train, X_test = df_small.iloc[train_index], df_small.iloc[test_index]

    # print("test: ", pd.to_datetime(X_test.index.to_series()).dt.day.unique())
    # print("training: ", pd.to_datetime(X_train.index.to_series()).dt.day.unique())

    # scarto nel training dei campioni di "classe 0", ne mantengo la frazione frac
    frac = 0.7
    X_train.drop(X_train.query('dishwasher == 0 & oven == 0 & wahing_machine == 0').sample(frac=frac).index,
                 inplace=True)

    # prendo le labels di training per i rispettivi classificatori
    y_train_washing_machine = X_train['wahing_machine']
    y_train_dishwasher = X_train['dishwasher']
    y_train_oven = X_train['oven']

    # prendo le labels di test per i rispettivi classificatori
    y_test_washing_machine = X_test['wahing_machine']
    y_test_dishwasher = X_test['dishwasher']
    y_test_oven = X_test['oven']

    # tolgo le labels e i datetime dai dataset di training e test
    X_train = X_train.drop(['DateTime', 'wahing_machine', 'dishwasher', 'oven'], axis=1)
    X_test = X_test.drop(['DateTime', 'wahing_machine', 'dishwasher', 'oven'], axis=1)

    # applico una normalizzazione min-max ai dati
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_norm = pd.DataFrame(scaler.transform(X_train),
                                columns=['ActivePower', 'ReactivePower', 'Voltage', 'Current', 'harmonic1_Real',
                                         'harmonic1_Imaginary', 'harmonic3_Real', 'harmonic3_Imaginary',
                                         'harmonic5_Real', 'harmonic5_Imaginary'])
    X_test_norm = pd.DataFrame(scaler.transform(X_test),
                               columns=['ActivePower', 'ReactivePower', 'Voltage', 'Current', 'harmonic1_Real',
                                        'harmonic1_Imaginary', 'harmonic3_Real', 'harmonic3_Imaginary',
                                        'harmonic5_Real', 'harmonic5_Imaginary'])

    '''
    ## MLP per la lavatrice
    print('MLP')
    mlp_washing_machine = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500, alpha=0.0001, solver='adam',
                                        verbose=10, random_state=21, tol=0.000000001)
    mlp_washing_machine.fit(X_train_norm, y_train_washing_machine)
    y_pred_washing_machine_mlp = mlp_washing_machine.predict(X_test_norm)
    accuracy_wm_mlp = accuracy_score(y_test_washing_machine, y_pred_washing_machine_mlp)
    precision_wm_mlp = precision_score(y_test_washing_machine, y_pred_washing_machine_mlp)
    recall_wm_mlp = recall_score(y_test_washing_machine, y_pred_washing_machine_mlp)
    '''

    '''
    # Dato che sulla washing machine abbiamo difficoltà a raggiungere un buon risultato, proviamo a usare l'adaboost
    ## Adaboost per la lavatrice
    print('Adaboost')
    adaboost_washing_machine = AdaBoostClassifier(n_estimators=500, random_state=0)
    adaboost_washing_machine.fit(X_train_norm, y_train_washing_machine)
    y_pred_washing_machine_ada = adaboost_washing_machine.predict(X_test_norm)
    accuracy_wm_ada = accuracy_score(y_test_washing_machine, y_pred_washing_machine_ada)
    precision_wm_ada = precision_score(y_test_washing_machine, y_pred_washing_machine_ada)
    recall_wm_ada = recall_score(y_test_washing_machine, y_pred_washing_machine_ada)
    '''

    # KNN
    # lavatrice
    knn_washing_machine = KNeighborsClassifier(n_neighbors=5)
    knn_washing_machine.fit(X_train_norm, y_train_washing_machine)
    y_pred_washing_machine = knn_washing_machine.predict(X_test_norm)
    accuracy_wm = accuracy_score(y_test_washing_machine, y_pred_washing_machine)
    precision_wm = precision_score(y_test_washing_machine, y_pred_washing_machine) if sum(
        y_pred_washing_machine) != 0 else None
    recall_wm = recall_score(y_test_washing_machine, y_pred_washing_machine) if sum(
        y_test_washing_machine) != 0 else None
    # lavastoviglie
    knn_dishwasher = KNeighborsClassifier(n_neighbors=5)
    knn_dishwasher.fit(X_train_norm, y_train_dishwasher)
    y_pred_dishwasher = knn_dishwasher.predict(X_test_norm)
    accuracy_dw = accuracy_score(y_test_dishwasher, y_pred_dishwasher)
    precision_dw = precision_score(y_test_dishwasher, y_pred_dishwasher) if sum(
        y_pred_dishwasher) != 0 else None
    recall_dw = recall_score(y_test_dishwasher, y_pred_dishwasher) if sum(
        y_test_dishwasher) != 0 else None
    # forno
    knn_oven = KNeighborsClassifier(n_neighbors=5)
    knn_oven.fit(X_train_norm, y_train_oven)
    y_pred_oven = knn_oven.predict(X_test_norm)
    accuracy_ov = accuracy_score(y_test_oven, y_pred_oven)
    precision_ov = precision_score(y_test_oven, y_pred_oven) if sum(
        y_pred_oven) != 0 else None
    recall_ov = recall_score(y_test_oven, y_pred_oven) if sum(
        y_test_oven) != 0 else None

    '''
    # Decision Tree
    print('DT')
    # lavatrice
    dt_washing_machine = DecisionTreeClassifier()
    dt_washing_machine.fit(X_train_norm, y_train_washing_machine)
    y_pred_washing_machine_DT = dt_washing_machine.predict(X_test_norm)
    accuracy_wm_DT = accuracy_score(y_test_washing_machine, y_pred_washing_machine_DT)
    precision_wm_DT = precision_score(y_test_washing_machine, y_pred_washing_machine_DT)
    recall_wm_DT = recall_score(y_test_washing_machine, y_pred_washing_machine_DT)
    # lavastoviglie
    dt_dishwasher = DecisionTreeClassifier()
    dt_dishwasher.fit(X_train_norm, y_train_dishwasher)
    y_pred_dishwasher_DT = dt_dishwasher.predict(X_test_norm)
    accuracy_dw_DT = accuracy_score(y_test_dishwasher, y_pred_dishwasher_DT)
    precision_dw_DT = precision_score(y_test_dishwasher, y_pred_dishwasher_DT)
    recall_dw_DT = recall_score(y_test_dishwasher, y_pred_dishwasher_DT)
    # forno
    dt_oven = DecisionTreeClassifier()
    dt_oven.fit(X_train_norm, y_train_oven)
    y_pred_oven_DT = dt_oven.predict(X_test_norm)
    accuracy_ov_DT = accuracy_score(y_test_oven, y_pred_oven_DT)
    precision_ov_DT = precision_score(y_test_oven, y_pred_oven_DT)
    recall_ov_DT = recall_score(y_test_oven, y_pred_oven_DT)
    '''

    # Calcolo i risultati per la fold in questione
    # KNN
    results_KNN = results_KNN._append(
        {'Class': 'washing_machine', 'Accuracy': accuracy_wm, 'Precision': precision_wm, 'Recall': recall_wm},
        ignore_index=True)
    results_KNN = results_KNN._append(
        {'Class': 'dishwasher', 'Accuracy': accuracy_dw, 'Precision': precision_dw, 'Recall': recall_dw},
        ignore_index=True)
    results_KNN = results_KNN._append(
        {'Class': 'oven', 'Accuracy': accuracy_ov, 'Precision': precision_ov, 'Recall': recall_ov}, ignore_index=True)

    '''
    # Decision Tree
    results_DT = results_DT.append(
        {'Class': 'washing_machine', 'Accuracy': accuracy_wm_DT, 'Precision': precision_wm_DT, 'Recall': recall_wm_DT},
        ignore_index=True)
    results_DT = results_DT.append(
        {'Class': 'dishwasher', 'Accuracy': accuracy_dw_DT, 'Precision': precision_dw_DT, 'Recall': recall_dw_DT},
        ignore_index=True)
    results_DT = results_DT.append(
        {'Class': 'oven', 'Accuracy': accuracy_ov_DT, 'Precision': precision_ov_DT, 'Recall': recall_ov_DT}, ignore_index=True)

    # Adaboost
    results_AdaBoost = results_AdaBoost.append(
        {'Class': 'washing_machine', 'Accuracy': accuracy_wm_ada, 'Precision': precision_wm_ada, 'Recall': recall_wm_ada},
        ignore_index=True)

    # MLP
    results_MLP = results_MLP.append(
        {'Class': 'washing_machine', 'Accuracy': accuracy_wm_mlp, 'Precision': precision_wm_mlp,
         'Recall': recall_wm_mlp},
        ignore_index=True)
    '''

winsound.Beep(1000, 1000)

# Printiamo i classification report dell'ultima fold
print('Classification report KNN washing machine')
print(classification_report(y_test_washing_machine, y_pred_washing_machine))
print('Classification report KNN dishwasher')
print(classification_report(y_test_dishwasher, y_pred_dishwasher))
print('Classification report KNN oven')
print(classification_report(y_test_oven, y_pred_oven))
'''
print('Classification report DT washing machine')
print(classification_report(y_test_washing_machine, y_pred_washing_machine_DT))
print('Classification report DT dishwasher')
print(classification_report(y_test_dishwasher, y_pred_dishwasher_DT))
print('Classification report DT oven')
print(classification_report(y_test_oven, y_pred_oven_DT))
print('Classification report AdaBoost washing machine')
print(classification_report(y_test_washing_machine, y_pred_washing_machine_ada))
print('Classification report MLP washing machine')
print(classification_report(y_test_washing_machine, y_pred_washing_machine_mlp))
'''

# aggiungo le F1 score per ogni fold
results_KNN['F1'] = 2 * (results_KNN['Precision'] * results_KNN['Recall']) / (
        results_KNN['Precision'] + results_KNN['Recall'])
'''
results_DT['F1'] = 2 * (results_DT['Precision'] * results_DT['Recall']) / (
        results_DT['Precision'] + results_DT['Recall'])
results_AdaBoost['F1'] = 2 * (results_AdaBoost['Precision'] * results_AdaBoost['Recall']) / (
        results_AdaBoost['Precision'] + results_AdaBoost['Recall'])
results_MLP['F1'] = 2 * (results_MLP['Precision'] * results_MLP['Recall']) / (
        results_MLP['Precision'] + results_MLP['Recall'])
'''

# calcolo le metriche aggregate (medie sulle fold)
results_DT_mean = results_DT.groupby(['Class']).mean()
results_KNN_mean = results_KNN.groupby(['Class']).mean()
results_AdaBoost_mean = results_AdaBoost.groupby(['Class']).mean()
results_MLP_mean = results_MLP.groupby(['Class']).mean()

results_DT.to_csv('Results/Results_DT.csv', index_label=results_DT.index.name)
results_DT_mean.to_csv('Results/Results_DT_mean.csv', index_label=results_DT_mean.index.name)
results_KNN.to_csv('Results/Results_KNN.csv', index_label=results_KNN.index.name)
results_KNN_mean.to_csv('Results/Results_KNN_mean.csv', index_label=results_KNN_mean.index.name)
results_AdaBoost.to_csv('Results/Results_AdaBoost.csv', index_label=results_AdaBoost.index.name)

end = time.time()
print("Tempo: ", (end - init) / 60, " minuti")

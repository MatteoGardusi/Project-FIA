import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import random
import warnings
import joblib
from tkinter import filedialog
warnings.simplefilter(action='ignore')

# Seleziono il file CSV
file_path = filedialog.askopenfilename(title="Seleziona il file CSV", filetypes=[("File CSV", "*.csv")])

# Verificare se è stato selezionato un file
if file_path:
    df = pd.read_csv(file_path, sep=',')
else:
    raise Exception("Nessun file selezionato")

df_small = df[
    ['ActivePower', 'ReactivePower', 'Voltage', 'Current', 'harmonic1_Real', 'harmonic1_Imaginary',
     'harmonic3_Real',
     'harmonic3_Imaginary', 'harmonic5_Real', 'harmonic5_Imaginary', 'wahing_machine', 'dishwasher', 'oven']]

# Andiamo a trasformare le classi in 0 e 1, in quanto ci occuperemo di classificazione binaria
df_small.loc[:, 'wahing_machine'] = df_small['wahing_machine'].apply(lambda x: 1 if x > 0 else 0)
df_small.loc[:, 'dishwasher'] = df_small['dishwasher'].apply(lambda x: 1 if x > 0 else 0)
df_small.loc[:, 'oven'] = df_small['oven'].apply(lambda x: 1 if x > 0 else 0)

kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(df_small)
results_RF = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])
results_ME = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])
k = 1
init = time.time()
for train_index, test_index in kf.split(df_small):

    X_train, X_test = df_small.iloc[train_index], df_small.iloc[test_index]
    # Questa parte di codice serve ridurre la dimensione del dataset di training, in quanto la classe off è molto
    # più numerosa delle altre. Riducendo il dataset di training, si ottiene una leggera diminuzione delle performance
    # a fronte di un notevole risparmio di tempo di esecuzione
    '''
    X_train = X_train.drop(
        X_train[(X_train['wahing_machine'] == 0) & (X_train['dishwasher'] == 0) & (X_train['oven'] == 0)].sample(
            frac=0.6).index)
    '''
    y_train_washing_machine = X_train['wahing_machine']
    y_train_dishwasher = X_train['dishwasher']
    y_train_oven = X_train['oven']
    y_train_off = X_train.apply(
        lambda x: 1 if x['wahing_machine'] == 0 and x['dishwasher'] == 0 and x['oven'] == 0 else 0,
        axis=1)

    y_test_washing_machine = X_test['wahing_machine']
    y_test_dishwasher = X_test['dishwasher']
    y_test_oven = X_test['oven']
    y_test_off = X_test.apply(
        lambda x: 1 if x['wahing_machine'] == 0 and x['dishwasher'] == 0 and x['oven'] == 0 else 0,
        axis=1)

    X_train = X_train.drop(['wahing_machine', 'dishwasher', 'oven'], axis=1)
    X_test = X_test.drop(['wahing_machine', 'dishwasher', 'oven'], axis=1)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    print('Random Forest Washing Machine')
    rf_washing_machine = RandomForestClassifier(n_estimators=20, random_state=0)
    rf_washing_machine.fit(X_train_norm, y_train_washing_machine)
    y_pred_washing_machine = rf_washing_machine.predict(X_test_norm)
    accuracy_wm_rf = accuracy_score(y_test_washing_machine, y_pred_washing_machine)
    precision_wm_rf = precision_score(y_test_washing_machine, y_pred_washing_machine)
    recall_wm_rf = recall_score(y_test_washing_machine, y_pred_washing_machine)
    washing_machine_prob = rf_washing_machine.predict_proba(X_test_norm)

    print('Random Forest Dishwasher')
    rf_dishwasher = RandomForestClassifier(n_estimators=20, random_state=0)
    rf_dishwasher.fit(X_train_norm, y_train_dishwasher)
    y_pred_dishwasher = rf_dishwasher.predict(X_test_norm)
    accuracy_dw_rf = accuracy_score(y_test_dishwasher, y_pred_dishwasher)
    precision_dw_rf = precision_score(y_test_dishwasher, y_pred_dishwasher)
    recall_dw_rf = recall_score(y_test_dishwasher, y_pred_dishwasher)
    dishwasher_prob = rf_dishwasher.predict_proba(X_test_norm)

    print('Random Forest Oven')
    rf_oven = RandomForestClassifier(n_estimators=20, random_state=0)
    rf_oven.fit(X_train_norm, y_train_oven)
    y_pred_oven = rf_oven.predict(X_test_norm)
    accuracy_ov_rf = accuracy_score(y_test_oven, y_pred_oven)
    precision_ov_rf = precision_score(y_test_oven, y_pred_oven)
    recall_ov_rf = recall_score(y_test_oven, y_pred_oven)
    oven_prob = rf_oven.predict_proba(X_test_norm)

    print('Random Forest Off')
    rf_off = RandomForestClassifier(n_estimators=20, random_state=0)
    rf_off.fit(X_train_norm, y_train_off)
    y_pred_off = rf_off.predict(X_test_norm)
    accuracy_off_rf = accuracy_score(y_test_off, y_pred_off)
    precision_off_rf = precision_score(y_test_off, y_pred_off)
    recall_off_rf = recall_score(y_test_off, y_pred_off)
    off_prob = rf_off.predict_proba(X_test_norm)

    results_RF = results_RF._append({'Class': 'Washing Machine', 'Accuracy': accuracy_wm_rf,
                                    'Precision': precision_wm_rf, 'Recall': recall_wm_rf}, ignore_index=True)
    results_RF = results_RF._append({'Class': 'Dishwasher', 'Accuracy': accuracy_dw_rf,
                                    'Precision': precision_dw_rf, 'Recall': recall_dw_rf}, ignore_index=True)
    results_RF = results_RF._append({'Class': 'Oven', 'Accuracy': accuracy_ov_rf,
                                    'Precision': precision_ov_rf, 'Recall': recall_ov_rf}, ignore_index=True)
    results_RF = results_RF._append({'Class': 'Off', 'Accuracy': accuracy_off_rf,
                                    'Precision': precision_off_rf, 'Recall': recall_off_rf}, ignore_index=True)

    # Una volta aver addestrato i quattro modelli sulla classe di interesse, si procede a introdurre uno strato di
    # classificazione finale che, in base alle probabilità di appartenenza alle classi, assegna l'istanza alla classe
    # con probabilità maggiore.

    y_pred = pd.DataFrame({'Washing Machine': y_pred_washing_machine, 'Dishwasher': y_pred_dishwasher,
                           'Oven': y_pred_oven, 'Off': y_pred_off})

    y_pred_final = pd.DataFrame(np.zeros((len(y_pred), 4)), columns=['wahing_machine', 'dishwasher', 'oven', 'Off'])

    y_prob = pd.DataFrame({'Washing Machine': washing_machine_prob[:, 1], 'Dishwasher': dishwasher_prob[:, 1],
                           'Oven': oven_prob[:, 1], 'Off': off_prob[:, 1]})

    y_test = pd.DataFrame({'wahing_machine': y_test_washing_machine, 'dishwasher': y_test_dishwasher,
                           'oven': y_test_oven})
    y_test = y_test.reset_index(drop=True)

    for i in range(len(y_pred)):
        if y_pred.iloc[i].sum() == 1:
            y_pred_final.iloc[i] = y_pred.iloc[i]
        else:
            y_prob_row = y_prob.iloc[i]
            y_prob_row[y_prob_row == y_prob_row.max()] = 1
            y_prob_row[y_prob_row != 1] = 0
            if y_prob_row.sum() > 1:
                if y_prob_row['Off'] == 1:
                    y_prob_row['Washing Machine'] = 0
                    y_prob_row['Dishwasher'] = 0
                    y_prob_row['Oven'] = 0
                else:
                    print(i)
                    indici = y_prob_row[y_prob_row == 1].index
                    indice = random.choice(list(indici))
                    y_prob_row[y_prob_row == 1] = 0
                    y_prob_row[indice] = 1
            y_pred_final.iloc[i] = y_prob_row

    y_pred_final = y_pred_final.drop('Off', axis=1)

    accuracy_final = accuracy_score(y_test, y_pred_final)
    precision_final = precision_score(y_test, y_pred_final, average='macro')
    recall_final = recall_score(y_test, y_pred_final, average='macro')

    accuracy_wm = accuracy_score(y_test_washing_machine, y_pred_final['wahing_machine'])
    precision_wm = precision_score(y_test_washing_machine, y_pred_final['wahing_machine'])
    recall_wm = recall_score(y_test_washing_machine, y_pred_final['wahing_machine'])
    accuracy_dw = accuracy_score(y_test_dishwasher, y_pred_final['dishwasher'])
    precision_dw = precision_score(y_test_dishwasher, y_pred_final['dishwasher'])
    recall_dw = recall_score(y_test_dishwasher, y_pred_final['dishwasher'])
    accuracy_ov = accuracy_score(y_test_oven, y_pred_final['oven'])
    precision_ov = precision_score(y_test_oven, y_pred_final['oven'])
    recall_ov = recall_score(y_test_oven, y_pred_final['oven'])

    results_ME = results_ME._append({'Class': 'Total', 'Accuracy': accuracy_final,
                                    'Precision': precision_final, 'Recall': recall_final}, ignore_index=True)
    results_ME = results_ME._append({'Class': 'Washing Machine', 'Accuracy': accuracy_wm,
                                    'Precision': precision_wm, 'Recall': recall_wm}, ignore_index=True)
    results_ME = results_ME._append({'Class': 'Dishwasher', 'Accuracy': accuracy_dw,
                                    'Precision': precision_dw, 'Recall': recall_dw}, ignore_index=True)
    results_ME = results_ME._append({'Class': 'Oven', 'Accuracy': accuracy_ov,
                                    'Precision': precision_ov, 'Recall': recall_ov}, ignore_index=True)

    # Si salvano i modelli addestrati e lo scaler per la normalizzazione dei dati.
    joblib.dump(rf_washing_machine, 'Models/RF_washing_machine_' + str(k) + '.pkl')
    joblib.dump(rf_dishwasher, 'Models/RF_dishwasher_' + str(k) + '.pkl')
    joblib.dump(rf_oven, 'Models/RF_oven_' + str(k) + '.pkl')
    joblib.dump(rf_off, 'Models/RF_off_' + str(k) + '.pkl')
    joblib.dump(scaler, 'Models/Scaler_' + str(k) + '.pkl')
    k = k + 1
end = time.time()
print("Tempo: ", (end - init) / 60, " minuti")

results_RF['F1'] = 2 * (results_RF['Precision'] * results_RF['Recall']) / (
        results_RF['Precision'] + results_RF['Recall'])
results_ME['F1'] = 2 * (results_ME['Precision'] * results_ME['Recall']) / (
        results_ME['Precision'] + results_ME['Recall'])

results_RF_mean = results_RF.groupby(['Class']).mean()
results_ME_mean = results_ME.groupby(['Class']).mean()

results_RF_std = results_RF.groupby(['Class']).std()
results_ME_std = results_ME.groupby(['Class']).std()

results_RF_mean.insert(1, 'Accuracy std', results_RF_std['Accuracy'])
results_RF_mean.insert(3, 'Precision std', results_RF_std['Precision'])
results_RF_mean.insert(5, 'Recall std', results_RF_std['Recall'])
results_RF_mean.insert(7, 'F1 std', results_RF_std['F1'])

results_ME_mean.insert(1, 'Accuracy std', results_ME_std['Accuracy'])
results_ME_mean.insert(3, 'Precision std', results_ME_std['Precision'])
results_ME_mean.insert(5, 'Recall std', results_ME_std['Recall'])
results_ME_mean.insert(7, 'F1 std', results_ME_std['F1'])

results_RF.to_csv('Results/Results_RF.csv', index_label=results_RF.index.name)
results_RF_mean.to_csv('Results/Results_RF_mean.csv', index_label=results_RF_mean.index.name)
results_ME.to_csv('Results/Results_ME.csv', index_label=results_ME.index.name)
results_ME_mean.to_csv('Results/Results_ME_mean.csv', index_label=results_ME_mean.index.name)

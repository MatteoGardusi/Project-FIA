import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import random
import warnings
import joblib
from tkinter import filedialog
warnings.filterwarnings('ignore')

# Aprire la finestra di dialogo per la selezione del file
file_path = filedialog.askopenfilename(title="Seleziona il file CSV", filetypes=[("File CSV", "*.csv")])

# Verificare se è stato selezionato un file
if file_path:
    # Leggere il file CSV
    df = pd.read_csv(file_path, sep=',')
else:
    raise Exception("Nessun file selezionato")

df_small = df[
    ['ActivePower', 'ReactivePower', 'Voltage', 'Current', 'harmonic1_Real', 'harmonic1_Imaginary',
     'harmonic3_Real',
     'harmonic3_Imaginary', 'harmonic5_Real', 'harmonic5_Imaginary', 'wahing_machine', 'dishwasher', 'oven']]

df_small.loc[:, 'wahing_machine'] = df_small['wahing_machine'].apply(lambda x: 1 if x > 0 else 0)
df_small.loc[:, 'dishwasher'] = df_small['dishwasher'].apply(lambda x: 1 if x > 0 else 0)
df_small.loc[:, 'oven'] = df_small['oven'].apply(lambda x: 1 if x > 0 else 0)

washing_machine_models = []
dishwasher_models = []
oven_models = []
off_models = []
scaler = []

for i in range(1, 11):
    washing_machine_models.append(joblib.load('Models/RF_washing_machine_' + str(i) + '.pkl'))
    dishwasher_models.append(joblib.load('Models/RF_dishwasher_' + str(i) + '.pkl'))
    oven_models.append(joblib.load('Models/RF_oven_' + str(i) + '.pkl'))
    off_models.append(joblib.load('Models/RF_off_' + str(i) + '.pkl'))
    scaler.append(joblib.load('Models/scaler_' + str(i) + '.pkl'))

y_test = df_small[['wahing_machine', 'dishwasher', 'oven']]
y_test['off'] = y_test.apply(lambda x: 1 if x['wahing_machine'] == 0 and x['dishwasher'] == 0 and x['oven'] == 0 else 0, axis=1)
X_test = df_small.drop(['wahing_machine', 'dishwasher', 'oven'], axis=1)

results_ME = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])

washing_machine_predictions = pd.DataFrame(np.zeros((len(y_test), 10)))
dishwasher_predictions = pd.DataFrame(np.zeros((len(y_test), 10)))
oven_predictions = pd.DataFrame(np.zeros((len(y_test), 10)))
off_predictions = pd.DataFrame(np.zeros((len(y_test), 10)))

washing_machine_prob = pd.DataFrame(np.zeros((len(y_test), 10)))
dishwasher_prob = pd.DataFrame(np.zeros((len(y_test), 10)))
oven_prob = pd.DataFrame(np.zeros((len(y_test), 10)))
off_prob = pd.DataFrame(np.zeros((len(y_test), 10)))

j = 0
for i in range(0, 10):
    X_test_norm = scaler[i].transform(X_test)
    washing_machine_predictions[j] = washing_machine_models[i].predict(X_test_norm)
    washing_machine_prob[j] = washing_machine_models[i].predict_proba(X_test_norm)[:, 1]
    dishwasher_predictions[j] = dishwasher_models[i].predict(X_test_norm)
    dishwasher_prob[j] = dishwasher_models[i].predict_proba(X_test_norm)[:, 1]
    oven_predictions[j] = oven_models[i].predict(X_test_norm)
    oven_prob[j] = oven_models[i].predict_proba(X_test_norm)[:, 1]
    off_predictions[j] = off_models[i].predict(X_test_norm)
    off_prob[j] = off_models[i].predict_proba(X_test_norm)[:, 1]
    j += 1

washing_machine_final_predictions = pd.DataFrame(np.zeros((len(y_test), 1)))
dishwasher_final_predictions = pd.DataFrame(np.zeros((len(y_test), 1)))
oven_final_predictions = pd.DataFrame(np.zeros((len(y_test), 1)))
off_final_predictions = pd.DataFrame(np.zeros((len(y_test), 1)))

washing_machine_final_prob = pd.DataFrame(np.zeros((len(y_test), 1)))
dishwasher_final_prob = pd.DataFrame(np.zeros((len(y_test), 1)))
oven_final_prob = pd.DataFrame(np.zeros((len(y_test), 1)))
off_final_prob = pd.DataFrame(np.zeros((len(y_test), 1)))

for i in range(0, len(X_test)):
    washing_machine_final_predictions.iloc[i] = washing_machine_predictions.iloc[i].value_counts().index[0]
    dishwasher_final_predictions.iloc[i] = dishwasher_predictions.iloc[i].value_counts().index[0]
    oven_final_predictions.iloc[i] = oven_predictions.iloc[i].value_counts().index[0]
    off_final_predictions.iloc[i] = off_predictions.iloc[i].value_counts().index[0]

    washing_machine_final_prob.iloc[i] = washing_machine_prob.iloc[i].mean()
    dishwasher_final_prob.iloc[i] = dishwasher_prob.iloc[i].mean()
    oven_final_prob.iloc[i] = oven_prob.iloc[i].mean()
    off_final_prob.iloc[i] = off_prob.iloc[i].mean()

y_pred = pd.concat([washing_machine_final_predictions, dishwasher_final_predictions, oven_final_predictions, off_final_predictions], axis=1)

y_pred_final = pd.DataFrame(np.zeros((len(y_pred), 4)), columns=['wahing_machine', 'dishwasher', 'oven', 'Off'])

y_prob = pd.concat([washing_machine_final_prob, dishwasher_final_prob, oven_final_prob, off_final_prob], axis=1)

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
                indice = random.choice(indici)
                y_prob_row[y_prob_row == 1] = 0
                y_prob_row[indice] = 1
        y_pred_final.iloc[i] = y_prob_row
y_pred_final = y_pred_final.drop('Off', axis=1)

y_test = y_test.drop('off', axis=1)
accuracy_final = accuracy_score(y_test, y_pred_final)
precision_final = precision_score(y_test, y_pred_final, average='macro')
recall_final = recall_score(y_test, y_pred_final, average='macro')

accuracy_wm = accuracy_score(y_test['wahing_machine'], y_pred_final['wahing_machine'])
precision_wm = precision_score(y_test['wahing_machine'], y_pred_final['wahing_machine'])
recall_wm = recall_score(y_test['wahing_machine'], y_pred_final['wahing_machine'])

accuracy_dw = accuracy_score(y_test['dishwasher'], y_pred_final['dishwasher'])
precision_dw = precision_score(y_test['dishwasher'], y_pred_final['dishwasher'])
recall_dw = recall_score(y_test['dishwasher'], y_pred_final['dishwasher'])

accuracy_ov = accuracy_score(y_test['oven'], y_pred_final['oven'])
precision_ov = precision_score(y_test['oven'], y_pred_final['oven'])
recall_ov = recall_score(y_test['oven'], y_pred_final['oven'])

results_ME = results_ME.append({'Class': 'Total', 'Accuracy': accuracy_final,
                                'Precision': precision_final, 'Recall': recall_final}, ignore_index=True)
results_ME = results_ME.append({'Class': 'Washing Machine', 'Accuracy': accuracy_wm,
                                'Precision': precision_wm, 'Recall': recall_wm}, ignore_index=True)
results_ME = results_ME.append({'Class': 'Dishwasher', 'Accuracy': accuracy_dw,
                                'Precision': precision_dw, 'Recall': recall_dw}, ignore_index=True)
results_ME = results_ME.append({'Class': 'Oven', 'Accuracy': accuracy_ov,
                                'Precision': precision_ov, 'Recall': recall_ov}, ignore_index=True)

results_ME['F1'] = 2 * (results_ME['Precision'] * results_ME['Recall']) / (
        results_ME['Precision'] + results_ME['Recall'])

results_ME_mean = results_ME.groupby(['Class']).mean()

results_ME.to_csv('Results/Results_ME_final.csv', index_label=results_ME.index.name)
results_ME_mean.to_csv('Results/Results_ME_mean_final.csv', index_label=results_ME_mean.index.name)
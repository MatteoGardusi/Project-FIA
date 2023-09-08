import pandas as pd
import matplotlib as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
import os
import time

init = time.time()
print(os.getcwd())
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import winsound

# Carichiamo il CSV
df = pd.read_csv('nilm/nilm/anonimized/25day_dataset.csv', sep=',', header=0, index_col=0)
timestamps = df.index.tolist()
df_small = df[['ActivePower', 'ReactivePower', 'Voltage', 'Current', 'harmonic1_Real', 'harmonic1_Imaginary',
               'harmonic3_Real', 'harmonic3_Imaginary', 'wahing_machine', 'dishwasher', 'oven']]

# trasformiamo i valori delle 3 colonne target in 0 e 1
df_small['wahing_machine'] = df_small['wahing_machine'].apply(lambda x: 1 if x > 0 else 0)
df_small['dishwasher'] = df_small['dishwasher'].apply(lambda x: 1 if x > 0 else 0)
df_small['oven'] = df_small['oven'].apply(lambda x: 1 if x > 0 else 0)

kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(df_small)

results_KNN = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])
finito = False
for train_index, test_index in kf.split(df_small):
    X_train, X_test = df_small.iloc[train_index], df_small.iloc[test_index]

    y_train_washing_machine = X_train['wahing_machine']
    y_train_dishwasher = X_train['dishwasher']
    y_train_oven = X_train['oven']

    y_test_washing_machine = X_test['wahing_machine']
    y_test_dishwasher = X_test['dishwasher']
    y_test_oven = X_test['oven']

    X_train = X_train.drop(['wahing_machine', 'dishwasher', 'oven'], axis=1)
    X_test = X_test.drop(['wahing_machine', 'dishwasher', 'oven'], axis=1)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # KNN per la lavatrice
    knn_washing_machine = KNeighborsClassifier(n_neighbors=5)
    knn_washing_machine.fit(X_train_norm, y_train_washing_machine)
    y_pred_washing_machine = knn_washing_machine.predict(X_test_norm)
    accuracy_wm = accuracy_score(y_test_washing_machine, y_pred_washing_machine)
    precision_wm = precision_score(y_test_washing_machine, y_pred_washing_machine)
    recall_wm = recall_score(y_test_washing_machine, y_pred_washing_machine)

    # KNN per la lavastoviglie
    knn_dishwasher = KNeighborsClassifier(n_neighbors=5)
    knn_dishwasher.fit(X_train_norm, y_train_dishwasher)
    y_pred_dishwasher = knn_dishwasher.predict(X_test_norm)
    accuracy_dw = accuracy_score(y_test_dishwasher, y_pred_dishwasher)
    precision_dw = precision_score(y_test_dishwasher, y_pred_dishwasher)
    recall_dw = recall_score(y_test_dishwasher, y_pred_dishwasher)

    # KNN per il forno
    knn_oven = KNeighborsClassifier(n_neighbors=5)
    knn_oven.fit(X_train_norm, y_train_oven)
    y_pred_oven = knn_oven.predict(X_test_norm)
    accuracy_ov = accuracy_score(y_test_oven, y_pred_oven)
    precision_ov = precision_score(y_test_oven, y_pred_oven)
    recall_ov = recall_score(y_test_oven, y_pred_oven)

    results_KNN = results_KNN.append(
        {'Class': 'washing_machine', 'Accuracy': accuracy_wm, 'Precision': precision_wm, 'Recall': recall_wm},
        ignore_index=True)
    results_KNN = results_KNN.append(
        {'Class': 'dishwasher', 'Accuracy': accuracy_dw, 'Precision': precision_dw, 'Recall': recall_dw},
        ignore_index=True)
    results_KNN = results_KNN.append(
        {'Class': 'oven', 'Accuracy': accuracy_ov, 'Precision': precision_ov, 'Recall': recall_ov}, ignore_index=True)

if finito:
    winsound.Beep(1000, 1000)

results_KNN['F1'] = 2 * (results_KNN['Precision'] * results_KNN['Recall']) / (
        results_KNN['Precision'] + results_KNN['Recall'])
results_KNN_mean = results_KNN.groupby(['Class']).mean()

print(results_KNN)
print(results_KNN_mean)

end = time.time()
print("durata: ", end-init)

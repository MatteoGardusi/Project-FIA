import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
import os
print(os.getcwd())
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Carichiamo il CSV
df = pd.read_csv('nilm/nilm/anonimized/25day_dataset.csv', sep=',', header=0, index_col=0)
timestamps = df.index.tolist()
df_small = df[
    ['ActivePower', 'ReactivePower', 'Voltage', 'Current', 'harmonic1_Real', 'harmonic1_Imaginary', 'harmonic3_Real',
     'harmonic3_Imaginary', 'harmonic5_Real', 'harmonic5_Imaginary', 'wahing_machine', 'dishwasher', 'oven']]

# trasformiamo i valori delle 3 colonne target in 0 e 1
df_small['wahing_machine'] = df_small['wahing_machine'].apply(lambda x: 1 if x > 0 else 0)
df_small['dishwasher'] = df_small['dishwasher'].apply(lambda x: 1 if x > 0 else 0)
df_small['oven'] = df_small['oven'].apply(lambda x: 1 if x > 0 else 0)

from sklearn.cluster import KMeans
df_small_no_target = df_small.drop(['wahing_machine', 'dishwasher', 'oven'], axis=1)
kmeans = KMeans(n_clusters=4, random_state=0).fit(df_small_no_target)
df_small['cluster'] = kmeans.labels_
# Vediamo quanti elementi ci sono in ogni cluster
print(df_small['cluster'].value_counts())

def apply_lowpass_filter(data, cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

cutoff_frequency = 0.1  # Frequenza di taglio del filtro
order = 4  # Ordine del filtro

# Frequenza di campionamento (potrebbe essere necessario adattarlo al tuo dataset)
fs = 100  # Esempio: campionamento a 100 Hz

# Filtra i segnali utilizzando il filtro passa-basso
df_small['ActivePower'] = apply_lowpass_filter(df_small['ActivePower'], cutoff_frequency, fs, order)
df_small['ReactivePower'] = apply_lowpass_filter(df_small['ReactivePower'], cutoff_frequency, fs, order)
df_small['Voltage'] = apply_lowpass_filter(df_small['Voltage'], cutoff_frequency, fs, order)
df_small['Current'] = apply_lowpass_filter(df_small['Current'], cutoff_frequency, fs, order)

# Grafichiamo i dati con plotly, voglio vedere se il filtro ha funzionato, facciamo solo il primo giorno però,
# cioè i primi 86400 secondi
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
fig.write_html('plotly.html', auto_open=True)

days = []
for i in range(0, 25):
    # Aggiungiamo a days una tupla con il numero di campioni con wahing machine > 0, dishwasher > 0 e oven > 0
    days.append((df_small['wahing_machine'][i * 86400:(i + 1) * 86400].sum(),
                    df_small['dishwasher'][i * 86400:(i + 1) * 86400].sum(),
                    df_small['oven'][i * 86400:(i + 1) * 86400].sum()))
days_classes = pd.DataFrame(days, columns=['wahing_machine', 'dishwasher', 'oven'])

folds = []
folds.append([1, 8, 4, 10, 6])
days_classes = days_classes.drop([0, 7, 3, 9, 5])
folds.append([11, 7, 5, 16, 13])
days_classes = days_classes.drop([10, 6, 4, 15, 12])
folds.append([2, 19, 22, 14, 23])
days_classes = days_classes.drop([1, 18, 21, 13, 22])
folds.append([24, 20, 3, 15, 21])
days_classes = days_classes.drop([23, 19, 2, 14, 20])
folds.append([18, 17, 9, 12, 25])
days_classes = days_classes.drop([17, 16, 8, 11, 24])

df_small['day'] = 0
for i in range(0, 25):
    df_small['day'][i * 86400:(i + 1) * 86400] = i + 1

df_small = df_small.drop(
    df_small[(df_small['wahing_machine'] == 0) & (df_small['dishwasher'] == 0) & (df_small['oven'] == 0)].sample(
        n=1900000).index)


kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(df_small)

results_KNN = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])
results_SVM = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])
results_DT = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])
results_MLP = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])
results_AdaBoost = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])
results_RF = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall'])
on = Condition(
    num_rows=500000,
    column_values={'wahing_machine': 1}
)
off = Condition(
    num_rows=5000,
    column_values={'wahing_machine': 0}
)
synthesizer = CTGANSynthesizer.load(
    filepath='CTGAN.pkl'
)

for fold in folds:
    # prendiamo tutti i dati dei 5 giorni
    data = df_small[(df_small['day'] == fold[0]) | (df_small['day'] == fold[1]) | (df_small['day'] == fold[2]) | (
                df_small['day'] == fold[3]) | (df_small['day'] == fold[4])]
    # facciamo shuffle dei dati
    data = data.sample(frac=1)
    # prendiamo 1/5 dei dati per il test set e 4/5 per il test set ma in maniera stratificata
    train, test = train_test_split(data, test_size=0.2, stratify=data[[ 'dishwasher']])

    train = train.drop(['day'], axis=1)
    test = test.drop(['day'], axis=1)

    X_train = train.drop(['wahing_machine', 'dishwasher', 'oven'], axis=1)
    X_test = test.drop(['wahing_machine', 'dishwasher', 'oven'], axis=1)
    y_train_washing_machine = train['wahing_machine']
    y_test_washing_machine = test['wahing_machine']
    y_train_dishwasher = train['dishwasher']
    y_test_dishwasher = test['dishwasher']
    y_train_oven = train['oven']
    y_test_oven = test['oven']

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # KNN per la lavatrice
    print('KNN')
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

for train_index, test_index in kf.split(df_small):
    X_train, X_test = df_small.iloc[train_index], df_small.iloc[test_index]
    '''
    X_train = X_train.drop(
        X_train[(X_train['wahing_machine'] == 0) & (X_train['dishwasher'] == 0) & (X_train['oven'] == 0)].sample(
            frac=0.70).index)
    '''

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

    # Random Forest per la lavatrice
    print('Random Forest')
    rf_washing_machine = RandomForestClassifier(n_estimators=1000, random_state=0)
    rf_washing_machine.fit(X_train_norm, y_train_washing_machine)
    y_pred_washing_machine = rf_washing_machine.predict(X_test_norm)
    accuracy_wm_rf = accuracy_score(y_test_washing_machine, y_pred_washing_machine)
    precision_wm_rf = precision_score(y_test_washing_machine, y_pred_washing_machine)
    recall_wm_rf = recall_score(y_test_washing_machine, y_pred_washing_machine)
    '''
    
    # Random Forest per la lavastoviglie
    rf_dishwasher = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_dishwasher.fit(X_train_norm, y_train_dishwasher)
    y_pred_dishwasher = rf_dishwasher.predict(X_test_norm)
    accuracy_dw_rf = accuracy_score(y_test_dishwasher, y_pred_dishwasher)
    precision_dw_rf = precision_score(y_test_dishwasher, y_pred_dishwasher)
    recall_dw_rf = recall_score(y_test_dishwasher, y_pred_dishwasher)

    # Random Forest per il forno
    rf_oven = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_oven.fit(X_train_norm, y_train_oven)
    y_pred_oven = rf_oven.predict(X_test_norm)
    accuracy_ov_rf = accuracy_score(y_test_oven, y_pred_oven)
    precision_ov_rf = precision_score(y_test_oven, y_pred_oven)
    recall_ov_rf = recall_score(y_test_oven, y_pred_oven)
    
    # KNN per la lavatrice
    print('KNN')
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
    
    # Decision Tree
    print('DT')
    dt_washing_machine = DecisionTreeClassifier()
    dt_washing_machine.fit(X_train_norm, y_train_washing_machine)
    y_pred_washing_machine_DT = dt_washing_machine.predict(X_test_norm)
    accuracy_wm_DT = accuracy_score(y_test_washing_machine, y_pred_washing_machine_DT)
    precision_wm_DT = precision_score(y_test_washing_machine, y_pred_washing_machine_DT)
    recall_wm_DT = recall_score(y_test_washing_machine, y_pred_washing_machine_DT)

    dt_dishwasher = DecisionTreeClassifier()
    dt_dishwasher.fit(X_train_norm, y_train_dishwasher)
    y_pred_dishwasher_DT = dt_dishwasher.predict(X_test_norm)
    accuracy_dw_DT = accuracy_score(y_test_dishwasher, y_pred_dishwasher_DT)
    precision_dw_DT = precision_score(y_test_dishwasher, y_pred_dishwasher_DT)
    recall_dw_DT = recall_score(y_test_dishwasher, y_pred_dishwasher_DT)

    dt_oven = DecisionTreeClassifier()
    dt_oven.fit(X_train_norm, y_train_oven)
    y_pred_oven_DT = dt_oven.predict(X_test_norm)
    accuracy_ov_DT = accuracy_score(y_test_oven, y_pred_oven_DT)
    precision_ov_DT = precision_score(y_test_oven, y_pred_oven_DT)
    recall_ov_DT = recall_score(y_test_oven, y_pred_oven_DT)

    results_DT = results_DT.append(
        {'Class': 'washing_machine', 'Accuracy': accuracy_wm_DT, 'Precision': precision_wm_DT, 'Recall': recall_wm_DT},
        ignore_index=True)
    results_DT = results_DT.append(
        {'Class': 'dishwasher', 'Accuracy': accuracy_dw_DT, 'Precision': precision_dw_DT, 'Recall': recall_dw_DT},
        ignore_index=True)
    results_DT = results_DT.append(
        {'Class': 'oven', 'Accuracy': accuracy_ov_DT, 'Precision': precision_ov_DT, 'Recall': recall_ov_DT},
        ignore_index=True)

    results_KNN = results_KNN.append(
        {'Class': 'washing_machine', 'Accuracy': accuracy_wm, 'Precision': precision_wm, 'Recall': recall_wm},
        ignore_index=True)
    results_KNN = results_KNN.append(
        {'Class': 'dishwasher', 'Accuracy': accuracy_dw, 'Precision': precision_dw, 'Recall': recall_dw},
        ignore_index=True)
    results_KNN = results_KNN.append(
        {'Class': 'oven', 'Accuracy': accuracy_ov, 'Precision': precision_ov, 'Recall': recall_ov}, ignore_index=True)
    '''
    results_RF = results_RF.append(
        {'Class': 'washing_machine', 'Accuracy': accuracy_wm_rf, 'Precision': precision_wm_rf, 'Recall': recall_wm_rf},
        ignore_index=True)
    '''
    results_RF = results_RF.append(
        {'Class': 'dishwasher', 'Accuracy': accuracy_dw_rf, 'Precision': precision_dw_rf, 'Recall': recall_dw_rf},
        ignore_index=True)
    results_RF = results_RF.append(
        {'Class': 'oven', 'Accuracy': accuracy_ov_rf, 'Precision': precision_ov_rf, 'Recall': recall_ov_rf},
        ignore_index=True)
    '''

results_DT['F1'] = 2 * (results_DT['Precision'] * results_DT['Recall']) / (
        results_DT['Precision'] + results_DT['Recall'])
results_KNN['F1'] = 2 * (results_KNN['Precision'] * results_KNN['Recall']) / (
        results_KNN['Precision'] + results_KNN['Recall'])
results_RF['F1'] = 2 * (results_RF['Precision'] * results_RF['Recall']) / (
        results_RF['Precision'] + results_RF['Recall'])

results_DT_mean = results_DT.groupby(['Class']).mean()
results_KNN_mean = results_KNN.groupby(['Class']).mean()
results_RF_mean = results_RF.groupby(['Class']).mean()


results_KNN.to_csv('Results/Results_KNN.csv', index_label=results_KNN.index.name)
results_KNN_mean.to_csv('Results/Results_KNN_mean.csv', index_label=results_KNN_mean.index.name)
results_DT.to_csv('Results/Results_DT.csv', index_label=results_DT.index.name)
results_DT_mean.to_csv('Results/Results_DT_mean.csv', index_label=results_DT_mean.index.name)
results_RF.to_csv('Results/Results_RF.csv', index_label=results_RF.index.name)
results_RF_mean.to_csv('Results/Results_RF_mean.csv', index_label=results_RF_mean.index.name)





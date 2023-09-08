import pandas as pd

df = pd.read_csv('nilm/anonimized/25day_dataset.csv', sep=',')

df_small = df[
    ['DateTime', 'wahing_machine', 'dishwasher', 'oven']]

# trasformiamo i valori delle 3 colonne target in 0 e 1
df_small.loc[:, 'wahing_machine'] = df_small['wahing_machine'].apply(lambda x: 1 if x > 0 else 0)
df_small.loc[:, 'dishwasher'] = df_small['dishwasher'].apply(lambda x: 1 if x > 0 else 0)
df_small.loc[:, 'oven'] = df_small['oven'].apply(lambda x: 1 if x > 0 else 0)

# Crea un nuovo DataFrame con una riga x giorno righe e 4 colonne
df_small['DateTime'] = pd.to_datetime(df_small['DateTime'])
df_small['DateTime'] = df_small['DateTime'].dt.date

# raggruppamento per la colonna 'DateTime' e somma tutte le altre colonne
result = df_small.groupby('DateTime').sum().reset_index()

# funzione lambda per la trasformazione
transform_function = lambda x: 1 if x > 0 else 0

# Applica la funzione alle colonne 'wahing_machine', 'dishwasher', 'oven'
result[['wahing_machine', 'dishwasher', 'oven']] = result[['wahing_machine', 'dishwasher', 'oven']].applymap(transform_function)

# Resetta l'indice del nuovo DataFrame
result.reset_index(drop=True, inplace=True)

# Salva il nuovo DataFrame in un file CSV
result.to_csv('daily_report.csv', index=False)
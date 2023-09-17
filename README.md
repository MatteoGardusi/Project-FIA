# Sistema di Riconoscimento delle Apparecchiature Elettriche (NILM)

## Descrizione del Progetto

Questo progetto si pone come obiettivo il riconoscimento di apparecchiature elettriche utilizzando un modello multiesperto. Il progetto è suddiviso in due file principali:

1. **Training dei Modelli** (`multiexpert.py`):
   - Questo programma si occupa dell'addestramento dei modelli multiesperto e dell'archiviazione di questi ultimi. I modelli addestrati saranno utilizzati per la classificazione.

2. **Classificazione e Valutazione** (`Classifiers.py`):
   - Questo programma prende i modelli già addestrati, costruisce i classificatori e misura le performance su un dataset di test per valutare l'efficacia del riconoscimento delle apparecchiature.

## Dataset

Il dataset deve essere strutturato come segue:

### Struttura del Dataset per il Training

- Il dataset deve contenere i seguenti dati: `['ActivePower', 'ReactivePower', 'Voltage', 'Current', 'harmonic1_Real', 'harmonic1_Imaginary',
     'harmonic3_Real',
     'harmonic3_Imaginary', 'harmonic5_Real', 'harmonic5_Imaginary', 'wahing_machine', 'dishwasher', 'oven']`.
- È consigliato avere un file di metadati o un file CSV.

### Struttura del Dataset per il Test

- Il dataset di test deve essere strutturato in modo simile al dataset di training, in particolare deve contenere le seguenti informazioni: `['ActivePower', 'ReactivePower', 'Voltage', 'Current', 'harmonic1_Real', 'harmonic1_Imaginary',
     'harmonic3_Real',
     'harmonic3_Imaginary', 'harmonic5_Real', 'harmonic5_Imaginary']`.

## Istruzioni per l'Esecuzione

Per eseguire in modo corretto i programmi, seguire questi passaggi:

1. **Training dei Modelli:**

    Esegui `multiexpert.py` per addestrare i modelli multiesperto utilizzando il dataset di training.
    ```
    python multiexpert.py
    ```
2. **Classificazione e Valutazione:**
    Esegui `Classifiers.py` per costruire i classificatori utilizzando i modelli addestrati e misurare le performance sui dati di test.
    ```
    python Classifiers.py
    ```
## Albero del Filesystem

Per la corretta esecuzione del programma è necessario che la directory contenente il codice del progetto abbia la seguente struttura:

```
.
└── Project_FIA
    ├── multiexpert.py
    ├── Classifiers.py
    ├── daily_count.csv
    ├── Results
        └── ...
    ├── Models
        └── ...
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```
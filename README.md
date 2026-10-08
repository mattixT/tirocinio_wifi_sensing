# Wi-Fi HAR dall'ampiezza CSI — NTU-Fi HAR

Codice del tirocinio *Wi-Fi Human Activity Recognition from CSI Amplitude Signals*
(Laurea triennale in Informatica, Sapienza Università di Roma — relatore Prof. Danilo Avola).

Il progetto riproduce in modo indipendente le baseline del benchmark **SenseFi** sul dataset
**NTU-Fi HAR** (6 attività: box, circle, clean, fall, run, walk), usando solo l'ampiezza della CSI
e tre modelli: **CNN-5**, **GRU** e **LSTM**. Rispetto a SenseFi aggiunge 5 seed per modello,
l'accuratezza esatta, la macro-F1, le matrici di confusione, le curve di apprendimento,
un'analisi degli errori file per file e due varianti con insieme di validazione.

## Riferimenti

- **SenseFi**: J. Yang et al., *SenseFi: A library and benchmark on deep-learning-empowered WiFi
  human sensing*, Patterns, 2023 (arXiv 2207.07859). Codice ufficiale:
  https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
- **NTU-Fi HAR**: J. Yang et al., *EfficientFi: Toward Large-Scale Lightweight WiFi Sensing via CSI
  Compression*, IEEE Internet of Things Journal, 2022 (arXiv 2204.04138).

Il preprocessing, le architetture dei modelli e il protocollo di addestramento sono quelli del
codice ufficiale di SenseFi. Il codice di questo repository è stato riscritto a partire da quello
originale e ne è stata verificata l'equivalenza: il preprocessing produce gli stessi valori
(differenza massima nulla su 48 file) e i modelli hanno lo stesso numero di parametri.

## Installazione

Gli esperimenti sono stati eseguiti con **Python 3.9.6** e le versioni esatte in `requirements.txt`.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Prima di lanciare qualunque script, attivare sempre l'ambiente (`source .venv/bin/activate`):
con un'altra installazione di Python i risultati potrebbero non essere identici.

## Dati e cartelle

Gli script usano percorsi assoluti a partire dalla cartella del tirocinio:

    TIROCINIO_TESI/
    ├── 02_CODICE/tirocinio_wifi_sensing/   ← questo repository
    ├── 03_DATASET/NTU-Fi_HAR/
    │   ├── train_amp/<attività>/*.mat      ← 936 campioni (156 per attività)
    │   └── test_amp/<attività>/*.mat       ← 264 campioni (44 per attività)
    ├── 04_ESPERIMENTI/                      ← una sottocartella per ogni run (creata da train.py)
    └── 05_FIGURE/                           ← figure generate dagli script

Il dataset si scarica seguendo le indicazioni del repository di SenseFi e non è incluso qui
(i file `.mat` e i pesi `.pt` sono esclusi da `.gitignore`). Per usare un'altra cartella, cambiare
le costanti `BASE`, `DATASET`, `EXPERIMENTS` e `FIGURE` all'inizio degli script.

Ogni file `.mat` contiene la variabile `CSIamp` di forma 342 × 2000: 3 coppie di antenne × 114
sottoportanti, 2000 pacchetti a 500 Hz (4 secondi). L'etichetta è data dalla cartella.

## Struttura del repository

    ├── preprocessing.py     preprocessing di un campione (identico a SenseFi)
    ├── data.py              caricamento dei dati, Dataset/DataLoader, insiemi di validazione
    ├── models.py            CNN-5 (NTU_Fi_LeNet), GRU, LSTM
    ├── train.py             addestramento e test di un modello con un seed
    ├── evaluate.py          ricarica un run salvato: verifica di coerenza e metriche per classe
    ├── analysis/            analisi dei risultati (leggono solo 04_ESPERIMENTI)
    │   ├── summary.py           media ± dev. std. sui seed, matrice di confusione sommata, curve medie
    │   ├── stats_report.py      tutti i numeri della relazione (test statistici, metriche per classe, ...)
    │   ├── error_analysis.py    file di test sbagliati e classi predette
    │   ├── errors_by_file.py    figura: errori per numero di file di test
    │   └── compare_files.py     figura: heatmap di file sbagliati e corretti
    ├── exploration/         esplorazione del dataset
    │   ├── csi_view.py          heatmap di un singolo campione
    │   ├── compare_samples.py   6 campioni della stessa attività
    │   ├── activity_samples.py  un campione per attività
    │   └── dataset_analysis.py  statistiche dell'insieme di addestramento
    ├── docs/scripts.md      guida dettagliata a tutti gli script
    ├── docs/notes.md        appunti sul dataset
    └── requirements.txt

I file della pipeline (`preprocessing`, `data`, `models`, `train`, `evaluate`) restano nella
cartella principale perché si importano a vicenda. Tutti gli script vanno lanciati **dalla
cartella principale del repository**.

La descrizione completa di ogni script (input, output, argomenti e uso nella relazione) è in
[`docs/scripts.md`](docs/scripts.md).

## Protocollo

**Baseline (fedele a SenseFi)**

| Elemento | Scelta |
|---|---|
| Preprocessing | un pacchetto ogni 4 (2000 → 500), normalizzazione (x − 42,3199) / 4,9802, forma (3, 114, 500), float32 |
| Addestramento | tutti i 936 campioni di train, nessuna validazione |
| Ottimizzazione | Adam, learning rate 1e-3, cross-entropy, batch 64, 30 epoche |
| Test | una sola volta, dopo l'ultima epoca, sui 264 campioni di test |
| Ripetizioni | 5 seed (0–4) per modello |

**Aggiunte di questo lavoro**

- accuratezza esatta (corretti / totale) accanto a quella calcolata come in SenseFi (media delle
  accuratezze dei batch, che sovrappesa l'ultimo batch da 8 campioni);
- loss e accuratezza sul test a ogni epoca, **solo per disegnare le curve**: non sono mai usate per
  scegliere l'epoca o gli iperparametri. Lo stato del generatore casuale viene salvato e
  ripristinato, così il monitoraggio non modifica l'addestramento;
- due varianti con insieme di validazione (non presenti in SenseFi):
  - `--val random`: 20% stratificato del train (748 train / 188 validazione), sempre gli stessi file;
  - `--val block`: i file numerati 140–159 di ogni attività (816 train / 120 validazione).

  In entrambe si registrano le curve di validazione e si testa anche il modello dell'epoca con la
  loss di validazione minima.

## Riprodurre i risultati

**1. Addestramento** (45 run in totale). GRU e LSTM sono più veloci su CPU.

    # baseline
    for s in 0 1 2 3 4; do python train.py --model cnn5 --seed $s; done
    for m in gru lstm; do for s in 0 1 2 3 4; do python train.py --model $m --seed $s --device cpu; done; done

    # varianti con validazione
    for v in random block; do
      for s in 0 1 2 3 4; do python train.py --model cnn5 --seed $s --val $v; done
      for m in gru lstm; do for s in 0 1 2 3 4; do python train.py --model $m --seed $s --device cpu --val $v; done; done
    done

Ogni run salva in `04_ESPERIMENTI/<modello>[_val-<modo>]_seed<N>/` i pesi (`model.pt`) e i
risultati (`results.json`).

**2. Riepiloghi e figure per modello**

    for m in cnn5 gru lstm; do python analysis/summary.py --model $m; done
    for v in random block; do for m in cnn5 gru lstm; do python analysis/summary.py --model $m --val $v; done; done

**3. Tutti i numeri della relazione** (salvati anche in `04_ESPERIMENTI/stats_report.txt`)

    python analysis/stats_report.py

**4. Analisi degli errori**

    python analysis/error_analysis.py --model gru
    python analysis/errors_by_file.py
    python analysis/compare_files.py --figure walk
    python analysis/compare_files.py --figure fall

**5. Esplorazione del dataset**

    python exploration/dataset_analysis.py
    python exploration/activity_samples.py
    python exploration/compare_samples.py      # attività scelta nella costante ACTIVITY
    python exploration/csi_view.py             # file scelto nella costante FILE

**6. Verifica di un singolo run**

    python evaluate.py --run gru_seed0 --device cpu

Con lo stesso seed e lo stesso dispositivo i risultati sono identici; su dispositivi diversi
(CPU, GPU Apple MPS, CUDA) possono differire leggermente.

## Contenuto di `results.json`

| Campo | Contenuto |
|---|---|
| `model`, `seed`, `device`, `epochs`, `learning_rate` | configurazione del run |
| `val`, `val_files` | variante di validazione e file usati (solo nelle varianti) |
| `history` | per ogni epoca: loss e accuratezza di train (`accuracy` alla SenseFi, `accuracy_exact`), `test_loss`, `test_accuracy`, e nelle varianti `val_loss`, `val_accuracy` |
| `test` | test dopo l'ultima epoca: `accuracy_exact`, `accuracy_sensefi`, `macro_f1`, `loss`, `confusion_matrix`, e per ogni campione `files`, `y_true`, `y_pred` |
| `test_best_val` | solo nelle varianti: lo stesso per il modello dell'epoca con loss di validazione minima, più `epoch` |

## Risultati principali

Accuratezza di test (%), media ± deviazione standard su 5 seed.

| Modello | Parametri | Baseline | Val random | Val block | SenseFi (1 run) |
|---|---|---|---|---|---|
| CNN-5 | 476 582 | 98,11 ± 1,52 | 96,52 ± 1,86 | 97,35 ± 1,14 | 98,70 |
| GRU   | 78 726  | 98,71 ± 0,63 | 98,71 ± 1,30 | 96,52 ± 0,94 | 97,66 |
| LSTM  | 104 838 | 97,88 ± 1,36 | 93,18 ± 5,42 | 97,80 ± 1,50 | 97,14 |

- I valori di SenseFi rientrano nella variabilità tra seed; le differenze tra i modelli della
  baseline non sono statisticamente significative.
- L'accuratezza calcolata come in SenseFi (media sui batch) può essere più bassa di oltre 10 punti
  di quella esatta in una singola esecuzione, perché l'ultimo batch del test (`walk192`–`walk199`)
  contiene alcuni dei file più difficili.
- Gli errori si concentrano nei file di test con numerazione più alta (blocco 190–199), per tutti
  i modelli.

## Note sul dataset

- La suddivisione train/test della versione distribuita da SenseFi non è casuale: per ogni attività
  il test contiene i file 16–19 e 160–199, il train i file 0–15 e 20–159.
- Due file del train hanno un nome anomalo (`circle20e.mat`, `clean40e.mat`), senza conseguenze.

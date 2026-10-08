# Guida agli script

Questa guida descrive uno per uno gli script del repository: a cosa servono, cosa leggono, cosa
producono e in quale parte della relazione vengono usati i loro risultati. Per installazione,
protocollo sperimentale e risultati principali si veda il [README](../README.md).

## Regole generali

- Tutti gli script si lanciano **dalla cartella principale del repository**, con l'ambiente attivo
  (`source .venv/bin/activate`), ad esempio `python analysis/summary.py --model gru`.
- I percorsi sono assoluti e partono da `/Users/mattiatagliola/Desktop/TIROCINIO_TESI`: sono definiti
  dalle costanti `BASE`, `DATASET`, `EXPERIMENTS` e `FIGURE` all'inizio di ogni file.
- Gli output sono di tre tipi: le cartelle dei run in `04_ESPERIMENTI/`, le figure in `05_FIGURE/` e
  il testo stampato a terminale (più il file `04_ESPERIMENTI/stats_report.txt`).
- Nessuno script modifica il dataset.
- Ogni file ha una docstring iniziale con scopo, output e comando di esecuzione: `python -c "import
  train; help(train)"` (o la semplice lettura del file) ne mostra il contenuto.

## Il flusso in breve

```
03_DATASET/NTU-Fi_HAR/  (file .mat)
   │
   ├──► exploration/*.py ─────────────────────────────► figure del dataset (capitolo 3)
   │
   └──► preprocessing.py ─► data.py ─► train.py ◄── models.py
                                          │
                                          ▼
                      04_ESPERIMENTI/<run>/  (results.json, model.pt)     45 run
                                          │
              ┌───────────────┬───────────┼────────────────┬──────────────────┐
              ▼               ▼           ▼                ▼                  ▼
         summary.py   stats_report.py  error_analysis.py  errors_by_file.py  compare_files.py
              │               │                            │                  │
              ▼               ▼                            ▼                  ▼
        curve e matrici   stats_report.txt         errori per numero   heatmap sbagliati
        di confusione     (tutti i numeri)         di file             vs corretti
                                          (capitoli 4 e 5)

evaluate.py: ricarica un run salvato e verifica che le metriche coincidano (controllo)
```

**Dipendenze tra i file.** Solo i file della cartella principale si importano a vicenda:
`data.py` usa `preprocessing.py`; `train.py` usa `data.py` e `models.py`; `evaluate.py` usa
`data.py`, `models.py` e `train.py`. Gli script di `analysis/` leggono soltanto i `results.json` (e,
per `compare_files.py`, i file `.mat` del test); quelli di `exploration/` leggono soltanto i file
`.mat`. Per questo possono essere eseguiti senza PyTorch e senza ripetere gli addestramenti.

## Tabella riassuntiva

| Script | Cosa fa | Legge | Scrive | Nella relazione |
|---|---|---|---|---|
| `preprocessing.py` | preprocessing di un campione (come SenseFi) | un file `.mat` | `preprocessing_test.png` | cap. 3, preprocessing |
| `data.py` | caricamento dei dati, DataLoader, validazione | dataset | — | cap. 4, protocolli |
| `models.py` | definizione di CNN-5, GRU e LSTM | — | — | cap. 4, modelli |
| `train.py` | addestramento e test di un run | dataset | `04_ESPERIMENTI/<run>/` | tutti i risultati |
| `evaluate.py` | verifica di un run salvato | un run, test | `<run>_training_curves.png` | non usato (controllo) |
| `analysis/summary.py` | riepilogo dei 5 seed di un modello | `results.json` | curve, matrici di confusione | cap. 5 |
| `analysis/stats_report.py` | tutti i numeri della relazione | `results.json` | `stats_report.txt` | tabelle del cap. 5 |
| `analysis/error_analysis.py` | elenco dei file di test sbagliati | `results.json` | solo terminale | cap. 5, errori |
| `analysis/errors_by_file.py` | errori per numero di file | `results.json` | `errori_per_numero_file*.png` | cap. 5, errori |
| `analysis/compare_files.py` | heatmap di file sbagliati e corretti | `results.json`, `.mat` di test | `<walk/fall>_errori_vs_corretti.png` | cap. 5, errori |
| `exploration/csi_view.py` | heatmap di un singolo campione | un file `.mat` | solo a schermo | — |
| `exploration/compare_samples.py` | 6 campioni della stessa attività | `.mat` di train | `confronto_<attività>.png` | cap. 3, appendice |
| `exploration/activity_samples.py` | un campione per attività | `.mat` di train | `campioni_per_attivita_<n>.png` | cap. 3 |
| `exploration/dataset_analysis.py` | statistiche dell'insieme di addestramento | `.mat` di train | `variazione_temporale_per_attivita.png` | cap. 3 |

---

## Pipeline (cartella principale)

### `preprocessing.py`

**A cosa serve.** Trasforma un file del dataset nell'input dei modelli, esattamente come il codice
ufficiale di SenseFi.

**Come funziona.** La funzione `load_sample(path)` legge la variabile `CSIamp` (matrice 342 × 2000:
3 coppie di antenne × 114 sottoportanti, 2000 pacchetti a 500 Hz) e applica tre passi:

1. sottocampionamento: un pacchetto ogni 4 (2000 → 500, la durata resta di 4 s);
2. normalizzazione con le costanti globali di SenseFi: `(x − 42,3199) / 4,9802`;
3. riorganizzazione in un tensore 3 × 114 × 500 di `float32`.

SenseFi esegue la normalizzazione prima del sottocampionamento: l'ordine non cambia i valori.

**Eseguito da solo** (`python preprocessing.py`) stampa forma, media e deviazione standard di un
campione prima e dopo il preprocessing e salva il confronto in `05_FIGURE/preprocessing_test.png`.

### `data.py`

**A cosa serve.** Carica l'intero dataset in memoria e prepara i `DataLoader` di PyTorch per
addestramento, validazione (facoltativa) e test.

**Funzioni principali.**

| Funzione | Ruolo |
|---|---|
| `load_split(split)` | carica e preprocessa tutti i file di `train_amp` o `test_amp`; restituisce `X` (N, 3, 114, 500) e `y` (N,) |
| `list_files(split)` | nomi dei file (`attività/file.mat`) nello stesso ordine di `load_split` |
| `split_validation(y)` | validazione casuale: 20% stratificato per attività, con seed fisso (`SPLIT_SEED = 0`) |
| `split_validation_block(files)` | validazione a blocchi: i file numerati 140–159 di ogni attività |
| `CSIDataset` | `Dataset` di PyTorch sui dati già in memoria; il test conserva anche il nome di ogni file |
| `get_loaders(val=...)` | restituisce `(train_loader, val_loader, test_loader)` con batch da 64 |

Le tre opzioni di `get_loaders`:

| `val` | Addestramento | Validazione | Test |
|---|---|---|---|
| `None` (baseline SenseFi) | 936 | — | 264 |
| `"random"` | 748 | 188 | 264 |
| `"block"` | 816 | 120 | 264 |

I file vengono letti in ordine fisso, per cui le etichette sono sempre 0 = box, 1 = circle,
2 = clean, 3 = fall, 4 = run, 5 = walk. Solo il loader di addestramento rimescola i dati.

**Eseguito da solo** (`python data.py`) stampa le dimensioni dei tre insiemi per le tre opzioni:
è un controllo rapido che la suddivisione sia quella attesa.

### `models.py`

**A cosa serve.** Definisce i tre modelli di SenseFi per NTU-Fi HAR, riprodotti senza modifiche.

| Classe | Nome da riga di comando | Struttura | Parametri |
|---|---|---|---|
| `NTU_Fi_LeNet` (CNN-5) | `cnn5` | 3 convoluzioni + 2 livelli completamente connessi | 476 582 |
| `NTU_Fi_GRU` | `gru` | GRU a un livello (ingresso 342, stato 64) + lineare 64 → 6 | 78 726 |
| `NTU_Fi_LSTM` | `lstm` | LSTM a un livello (ingresso 342, stato 64) + lineare 64 → 6 | 104 838 |

Le reti ricorrenti leggono il campione come una sequenza di 500 istanti, ciascuno con 342 valori, e
classificano a partire dallo stato nascosto dell'ultimo istante. Il dizionario `MODELS` collega il
nome usato in `train.py --model` alla classe.

**Eseguito da solo** (`python models.py`) stampa la forma dell'uscita di ogni livello della CNN-5 e
il numero di parametri dei tre modelli.

### `train.py`

**A cosa serve.** Addestra e testa **un** modello con **un** seed: ogni esecuzione è un "run".

**Argomenti.**

| Argomento | Valori | Significato |
|---|---|---|
| `--model` | `cnn5`, `gru`, `lstm` | modello da addestrare |
| `--seed` | intero (0–4 negli esperimenti) | inizializzazione casuale |
| `--device` | `auto`, `cpu`, `mps`, `cuda` | dispositivo di calcolo (GRU e LSTM sono più veloci su CPU) |
| `--val` | `none`, `random`, `block` | baseline SenseFi oppure una delle due varianti con validazione |

**Cosa fa, in ordine.**

1. fissa il seed di Python, NumPy e PyTorch (prima di creare il modello);
2. carica i dati con `get_loaders`;
3. addestra per 30 epoche con Adam (lr 10⁻³), cross-entropy e batch da 64;
4. a ogni epoca registra loss e accuratezza di addestramento (sia come media sui batch, alla SenseFi,
   sia esatta) e loss e accuratezza sul test, **solo per disegnare le curve**: lo stato del
   generatore casuale viene salvato e ripristinato, così il monitoraggio non modifica l'addestramento;
5. nelle varianti con validazione registra anche loss e accuratezza di validazione e conserva una
   copia dei pesi dell'epoca con la loss di validazione minima;
6. testa il modello dell'ultima epoca (risultato principale) e, nelle varianti, anche quello
   dell'epoca migliore sulla validazione;
7. salva tutto in `04_ESPERIMENTI/<modello>[_val-<modo>]_seed<N>/`.

**Output.** `model.pt` (pesi dell'ultima epoca), `model_best_val.pt` (solo nelle varianti) e
`results.json`, i cui campi sono descritti nel README. Il terminale mostra l'andamento epoca per
epoca e le metriche finali.

**Esempi.**

    python train.py --model cnn5 --seed 0
    python train.py --model gru --seed 3 --device cpu --val block

### `evaluate.py`

**A cosa serve.** È uno strumento di controllo: ricarica i pesi di un run salvato, ripete la
predizione sul test e verifica che l'accuratezza coincida con quella salvata da `train.py`
(stampa `OK` oppure `DIFFERENT`). Stampa poi precisione, richiamo e F1 per classe e la matrice di
confusione del run, e disegna le curve di addestramento (`05_FIGURE/<run>_training_curves.png`,
non usata nella relazione). Non addestra nulla; nelle varianti con validazione ricarica il modello
dell'ultima epoca.

    python evaluate.py --run gru_seed0 --device cpu

---

## Analisi dei risultati (`analysis/`)

Questi script leggono soltanto i run salvati in `04_ESPERIMENTI/`: si possono rieseguire in pochi
secondi senza ripetere gli addestramenti.

### `analysis/summary.py`

**A cosa serve.** Riassume i 5 run di un modello, per la baseline o per una variante.

**Cosa produce.**

- a terminale: accuratezza e macro-F1 di ogni run, media ± deviazione standard campionaria, matrice
  di confusione sommata; nelle varianti anche il test all'epoca scelta sulla validazione e le
  epoche scelte;
- `05_FIGURE/<modello>[_val-<modo>]_5runs_confusion_matrix.png`: matrice di confusione sommata sui
  5 run (1320 predizioni);
- `05_FIGURE/<modello>[_val-<modo>]_5runs_curves.png`: curve di loss (scala logaritmica) e
  accuratezza per addestramento, test e, se presente, validazione, come media sui run con una banda
  di ± 1 deviazione standard. Per la loss media e banda sono calcolate sui logaritmi (media
  geometrica), così la banda resta positiva sull'asse logaritmico.

**Esempi.**

    python analysis/summary.py --model cnn5
    python analysis/summary.py --model lstm --val block

### `analysis/stats_report.py`

**A cosa serve.** Ricalcola **tutti** i numeri riportati nella relazione a partire dai
`results.json` e li salva in `04_ESPERIMENTI/stats_report.txt`. Le sezioni del report:

1. accuratezza per seed, media ± deviazione standard, intervallo di confidenza al 95%, macro-F1
   (tutti i modelli e tutte le varianti);
2. confronto tra i modelli della baseline: test di Welch e di Mann–Whitney;
3. accuratezza esatta e accuratezza di SenseFi (media sui batch) per ogni run;
4. precisione, richiamo e F1 per classe dalle matrici di confusione sommate;
5. curve di apprendimento: accuratezza di test alle epoche 5, 10, 20 e 30, prima epoca con il 100%
   in addestramento, loss finali e la "migliore epoca scelta sul test" (stima ottimistica, solo
   come termine di confronto);
6. file di test sbagliati: file distinti, concentrazione nei blocchi 180–199 e 190–199, test
   binomiale;
7. varianti con validazione: confronto con la baseline, validazione rispetto al test, epoca scelta
   sulla validazione.

**Esempio.**

    python analysis/stats_report.py
    python analysis/stats_report.py --dir <altra cartella di run>

Nota: il file `04_ESPERIMENTI/statistics.txt` è stato prodotto dalla versione precedente dello
script (che si chiamava `statistics.py`); il contenuto è lo stesso di `stats_report.txt`.

### `analysis/error_analysis.py`

**A cosa serve.** Elenca, per un modello, i file di test classificati erroneamente almeno una volta
nei 5 run: per ciascuno indica in quanti run è sbagliato e con quale classe, dal più al meno
frequente. È il punto di partenza dell'analisi degli errori; produce solo testo.

    python analysis/error_analysis.py --model gru
    python analysis/error_analysis.py --model cnn5 --val block

### `analysis/errors_by_file.py`

**A cosa serve.** Mostra se gli errori dipendono dalla numerazione dei file. I file di test hanno
gli stessi 44 numeri per tutte le attività (16–19 e 160–199): per ogni numero la figura riporta il
totale degli errori, sommati sulle 6 attività e sui 5 run, con un pannello per modello e i blocchi
180–189 e 190–199 evidenziati.

**Output.** `05_FIGURE/errori_per_numero_file.png` (baseline) oppure
`05_FIGURE/errori_per_numero_file_val-<modo>.png`.

    python analysis/errors_by_file.py
    python analysis/errors_by_file.py --val block

### `analysis/compare_files.py`

**A cosa serve.** Confronta visivamente file sbagliati e file corretti. Ogni figura è una griglia
2 × 3 di heatmap (ampiezza grezza, prima coppia di antenne, stessa scala di colori): nella prima
riga due file corretti dell'attività reale e un file dell'attività con cui viene confusa, nella
seconda due file sbagliati e un altro file dell'attività confusa. I titoli sono calcolati dai
`results.json` della baseline ("sbagliato k/5 → classe" oppure "mai sbagliato").

| `--figure` | Modello | Coppia di attività |
|---|---|---|
| `walk` | CNN-5 | camminata scambiata per corsa |
| `fall` | GRU | caduta scambiata per pulizia del pavimento |

I sei file di ogni figura si scelgono nel dizionario `FIGURES` all'inizio dello script.

**Output.** `05_FIGURE/walk_errori_vs_corretti.png`, `05_FIGURE/fall_errori_vs_corretti.png`.

    python analysis/compare_files.py --figure walk

---

## Esplorazione del dataset (`exploration/`)

Questi script leggono direttamente i file `.mat` (ampiezza grezza, senza preprocessing) e servono a
conoscere il dataset prima di addestrare i modelli.

### `exploration/csi_view.py`

Mostra la heatmap di un singolo campione (prima coppia di antenne): sottoportanti sull'asse
verticale, pacchetti sull'asse orizzontale. Le bande orizzontali sono la parte statica del canale,
le tracce verticali il movimento della persona. Il file si sceglie con la costante `FILE`; la
figura viene solo mostrata, non salvata.

### `exploration/compare_samples.py`

Mette a confronto sei campioni di addestramento della stessa attività (file 0, 30, 60, 90, 120,
150) con la stessa scala di colori, per vedere quanto varia una stessa attività da una ripetizione
all'altra. L'attività si sceglie con la costante `ACTIVITY`.
**Output:** `05_FIGURE/confronto_<attività>.png`.

### `exploration/activity_samples.py`

Un campione per ciascuna delle 6 attività (stesso numero di file, insieme di addestramento), con la
stessa scala di colori: mostra che le attività lasciano "firme" diverse nella CSI.
**Output:** `05_FIGURE/campioni_per_attivita_<n>.png`.

    python exploration/activity_samples.py --sample 0

### `exploration/dataset_analysis.py`

Statistiche dell'insieme di addestramento (936 file):

1. controlla che tutti i file abbiano forma 342 × 2000;
2. calcola media e deviazione standard globali e le confronta con le costanti di SenseFi
   (risultato: 42,3040 e 4,9838 contro 42,3199 e 4,9802);
3. misura la variazione temporale di ogni file (deviazione standard nel tempo di ciascuna delle 342
   righe, mediata sulle righe) e la riassume per attività.

Usa solo l'insieme di addestramento, così nessuna informazione proviene dal test.
**Output:** `05_FIGURE/variazione_temporale_per_attivita.png`.

---

## Quale script produce quale figura

| Figura in `05_FIGURE/` | Script | Uso nella relazione |
|---|---|---|
| `preprocessing_test.png` | `preprocessing.py` | cap. 3, preprocessing |
| `campioni_per_attivita_<n>.png` | `exploration/activity_samples.py` | cap. 3, il campione come immagine |
| `variazione_temporale_per_attivita.png` | `exploration/dataset_analysis.py` | cap. 3, statistiche |
| `confronto_<attività>.png` | `exploration/compare_samples.py` | cap. 3 e appendice |
| `<modello>[_val-<modo>]_5runs_curves.png` | `analysis/summary.py` | cap. 5, curve e varianti |
| `<modello>[_val-<modo>]_5runs_confusion_matrix.png` | `analysis/summary.py` | cap. 5, matrici di confusione |
| `errori_per_numero_file[_val-<modo>].png` | `analysis/errors_by_file.py` | cap. 5 e appendice |
| `walk_errori_vs_corretti.png`, `fall_errori_vs_corretti.png` | `analysis/compare_files.py` | cap. 5, errori |
| `<run>_training_curves.png` | `evaluate.py` | non usata |

Nota sui nomi: le figure della baseline già presenti in `05_FIGURE/` si chiamano
`<modello>_curves_5runs.png` e `<modello>_confusion_matrix_5runs.png`, perché sono state generate
con una versione precedente di `summary.py`. Rieseguendo lo script si ottengono gli stessi grafici
con i nomi nuovi (`<modello>_5runs_curves.png`, `<modello>_5runs_confusion_matrix.png`).

## Riprodurre tutto da zero, in ordine

1. (facoltativo) esplorazione del dataset: gli script di `exploration/`;
2. i 45 addestramenti con `train.py` (comandi nel README);
3. `analysis/summary.py` per ogni modello, per la baseline e per le due varianti;
4. `analysis/stats_report.py` per tutti i numeri;
5. `analysis/error_analysis.py`, `analysis/errors_by_file.py` e `analysis/compare_files.py` per
   l'analisi degli errori;
6. (facoltativo) `evaluate.py` su qualche run, come verifica.

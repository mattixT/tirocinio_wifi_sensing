# Riconoscimento di attività umane dall'ampiezza del CSI Wi-Fi — NTU-Fi HAR

Codice della relazione di tirocinio *Riconoscimento di attività umane tramite l'ampiezza del CSI
Wi-Fi: riproduzione e analisi del protocollo di valutazione su NTU-Fi HAR* (Laurea triennale in
Informatica, Sapienza Università di Roma; relatore Prof. Danilo Avola).

Riproduce tre modelli del benchmark [SenseFi](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark)
(CNN-5, GRU, LSTM) sul dataset NTU-Fi HAR e ne analizza il protocollo di valutazione.

## Installazione

Python 3.9.6, con le versioni esatte di `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Gli script si lanciano dalla cartella principale del repository, con l'ambiente attivo.
Le figure di `work_figures/` richiedono anche LaTeX (pdflatex).

## Dati e cartelle

Il dataset non è incluso e si scarica seguendo le indicazioni del repository di SenseFi.
Gli script usano percorsi assoluti, definiti dalle costanti all'inizio di ogni file (`BASE`,
`DATASET`, `EXPERIMENTS`, `FIGURE`), da modificare se la cartella è diversa:

```
TIROCINIO_TESI/
├── 02_CODICE/tirocinio_wifi_sensing/   questo repository
├── 03_DATASET/NTU-Fi_HAR/              train_amp/ e test_amp/, una sottocartella per attività
├── 04_ESPERIMENTI/                     risultati delle esecuzioni
├── 05_FIGURE/                          figure degli script di analisi
└── 07_RELAZIONE/images/                figure della relazione
```

## Struttura

```
pipeline/              preprocessing, caricamento dei dati, modelli, addestramento
results_analysis/      riepiloghi, test statistici e analisi degli errori (leggono 04_ESPERIMENTI)
dataset_exploration/   statistiche e figure del dataset
utilities/             verifica che il preprocessing coincida con quello di SenseFi
work_figures/        figure della relazione (make_all.py le rigenera tutte)
```

## Esecuzione

**Addestramento.** Ogni comando addestra e testa un modello con un seed e salva pesi e
`results.json` in `04_ESPERIMENTI/<modello>[_val-<modo>][_norm-<modo>]_seed<N>/`.

```bash
python pipeline/train.py --model <cnn5|gru|lstm> --seed <0-4> [--device <auto|cpu|mps|cuda>]
                         [--val <none|random|block>] [--norm <sensefi|sample|subcarrier>]
```

Tutte le 75 esecuzioni della relazione (3 modelli × 5 configurazioni × 5 seed):

```bash
for m in cnn5 gru lstm; do
  if [ $m = cnn5 ]; then d=auto; else d=cpu; fi
  for s in 0 1 2 3 4; do
    python pipeline/train.py --model $m --seed $s --device $d
    for v in random block; do python pipeline/train.py --model $m --seed $s --device $d --val $v; done
    for n in sample subcarrier; do python pipeline/train.py --model $m --seed $s --device $d --norm $n; done
  done
done
```

Nella relazione la CNN-5 è stata addestrata sulla GPU Apple (`--device auto`), GRU e LSTM su CPU:
con lo stesso seed e lo stesso dispositivo i risultati sono identici, su dispositivi diversi possono
differire leggermente.

**Analisi dei risultati**

```bash
python results_analysis/summary.py --model <cnn5|gru|lstm> [--val ...] [--norm ...]   # riepilogo e figure di un modello
python results_analysis/stats_report.py                                              # tutti i numeri della relazione
python results_analysis/error_analysis.py --model <cnn5|gru|lstm>                    # campioni sbagliati
python results_analysis/errors_by_file.py                                            # figura: errori per numero di campione
python results_analysis/compare_files.py --figure walk                               # figura: campioni sbagliati e corretti
```

**Dataset e verifiche**

```bash
python dataset_exploration/dataset_analysis.py           # statistiche del dataset
python utilities/check_preprocessing.py                  # preprocessing identico a SenseFi
python pipeline/evaluate.py --run gru_seed0              # ricontrolla un'esecuzione salvata
```

**Figure della relazione**

```bash
python work_figures/make_all.py [--skip-slow]            # per evitare di leggere tutto il dataset
```

# Wi-Fi HAR da ampiezza CSI — NTU-Fi HAR

Pipeline per il tirocinio: riconoscimento di attività umane (box, circle, clean, fall, run, walk)
dall'ampiezza CSI del dataset NTU-Fi HAR, con modelli ripresi da SenseFi (CNN-5, GRU, LSTM).

## Installazione
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## File
- `data.py`    caricamento .mat, sottocampionamento temporale, normalizzazione, split train/val/test
- `models.py`  CNN5, GRU, LSTM (stessi parametri di SenseFi: 0.477M / 0.079M / 0.105M)
- `train.py`   addestramento + valutazione, salva tutto in `04_ESPERIMENTI/<run>/`
- `aggregate.py` tabella riassuntiva (media ± dev. std sui seed) in `04_ESPERIMENTI/summary.csv`
- `dataset.py` script di esplorazione iniziale (non usato dalla pipeline)

## Protocollo
- Train ufficiale (936) diviso 80/20 stratificato in train (748) / validazione (188).
- Epoca migliore scelta sulla macro-F1 di validazione; test ufficiale (264) usato una sola volta.
- Adam, lr 1e-3, batch 64, 30 epoche (come SenseFi per NTU-Fi). Più seed per media ± dev. std.
- Il primo run crea una cache in `03_DATASET/NTU-Fi_HAR/cache/` (~800 MB) per velocizzare i successivi.

## Esempi
    python train.py --model cnn5 --seed 0 --tag baseline
    for m in cnn5 gru lstm; do for s in 0 1 2 3 4; do python train.py --model $m --seed $s --tag baseline; done; done
    python train.py --model gru --norm zscore --tag norm           # analisi normalizzazione
    python train.py --model gru --step 8 --tag length              # T=250 invece di 500
    python aggregate.py

# Contesto per Claude — Tirocinio Wi-Fi HAR

Tirocinio di Laurea Triennale in Informatica (Sapienza), relatore Prof. Danilo Avola.
Relazione in **italiano**, firmata dal prof entro il **27 novembre 2026** (bozza al prof entro ~13 novembre).

## Tema concordato con il prof
Wi-Fi Human Activity Recognition from CSI Amplitude Signals
(*Riconoscimento di attività umane tramite segnali Wi-Fi CSI*).
- Dataset: **NTU-Fi HAR** (UT-HAR solo se emergono problemi tecnici). 6 classi: box, circle, clean, fall, run, walk.
- Solo **ampiezza** CSI, niente fase. Niente Person ID/Re-ID. Niente transformer o architetture complesse.
- Base di riferimento: benchmark **SenseFi** (`../../01_DOCUMENTI/SenseFi.pdf`).
- Da riportare: accuracy, macro-F1, confusion matrix, curve training/validation; discutere attività riconosciute meglio/peggio.
- Se resta tempo: effetto della normalizzazione, della lunghezza temporale del segnale, della complessità del modello.

## Struttura cartelle (radice: ~/Desktop/TIROCINIO_TESI)
- `02_CODICE/tirocinio_wifi_sensing/` questo repo (GitHub: mattixT/tirocinio_wifi_sensing)
- `03_DATASET/NTU-Fi_HAR/{train_amp,test_amp}/<classe>/*.mat` — 'CSIamp' (342, 2000); cache .npz in `cache/`
- `04_ESPERIMENTI/<run>/` output dei run; `summary.csv` da `aggregate.py`
- `05_FIGURE/` figure per la tesi; `07_BACKUP_TESI/` tesi LaTeX (template sapthesis)

## Codice
- `data.py` loader, sottocampionamento `step` (4 → T=500), normalizzazione `sensefi|zscore|subcarrier|none`, split
- `models.py` CNN5, GRU, LSTM (parametri identici a SenseFi: 0.477M / 0.079M / 0.105M)
- `train.py` training + valutazione; `aggregate.py` tabella media ± std sui seed
- `dataset.py`, `run.py`: script iniziali di esplorazione, non usati dalla pipeline

## Protocollo (decisione presa)
Validazione 80/20 stratificata ricavata dal train ufficiale (748/188); epoca scelta sulla macro-F1 di validazione;
test ufficiale (264) usato una sola volta; 5 seed → media ± dev. std. Adam lr 1e-3, batch 64, 30 epoche.
(SenseFi riporta la *best test accuracy*: il nostro protocollo è più corretto, è un punto da evidenziare.)

## Stato (25/09/2026)
Primi run seed 0: CNN-5 95.83% / GRU 97.35% / LSTM 96.21% acc test (SenseFi: 98.70 / 97.66 / 97.14).
Validazione ~99-100% ma test 96-97%: il test (file 16-19 e 160-199 di ogni classe) non è i.i.d. rispetto al train.
Errori ricorrenti: walk→run, circle→clean. File con nome anomalo: circle20e.mat, clean40e.mat (gestiti).
Prossimi passi: 5 seed × 3 modelli; analisi normalizzazione e lunghezza temporale; riscrivere la tesi (ora è
ancora su person identification e in inglese, bibliografia del template da sostituire).

## Preferenze
Spiegazioni concise ed esempi concreti; testo della tesi in italiano.

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
- `04_ESPERIMENTI/<run>/` output dei run
- `05_FIGURE/` figure per la tesi; `07_BACKUP_TESI/` tesi LaTeX (template sapthesis)

## Stato (25/09/2026) — RIPARTENZA DA ZERO
Lo studente ha chiesto di cancellare la pipeline scritta da Claude (data.py, models.py, train.py, aggregate.py;
recuperabile dal commit 1b4beca) per ricostruirla passo passo capendo ogni parte.
Nel repo restano solo `dataset.py` (esplorazione del dataset, scritto dallo studente) e `run.py` (vuoto).
Note utili già emerse: file con nome anomalo circle20e.mat, clean40e.mat; il test (file 16-19 e 160-199 di
ogni classe) non è i.i.d. rispetto al train. La tesi è ancora su person identification e in inglese.

## Preferenze
Spiegazioni concise ed esempi concreti; testo della tesi in italiano.
Procedere a piccoli passi: lo studente deve capire tutto. Una cosa alla volta, spiegata, prima di scrivere codice;
niente pipeline complete fatte in un colpo solo.

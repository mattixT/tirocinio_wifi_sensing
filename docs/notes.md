Il dataset NTU-Fi HAR contiene 6 classi: box, circle, clean, fall, run, walk. Il training set contiene 156 campioni per ogni attività, per un totale di 936 campioni, mentre il test set ne contiene 44 per classe, per un totale di 264 campioni. Il dataset è pertanto bilanciato rispetto alle classi, con complessivamente 200 campioni per attività e 1.200 campioni totali e ogni campione viene fornito al modello nella forma 3 x 114 x 500 (3 antenne x 114 sottocanali x 500 istanti temporali). Ogni file contiene un singolo campione CSI acquisito durante l’esecuzione di una determinata attività. L’etichetta del campione è ricavata dalla cartella in cui si trova il file, ad esempio `fall` oppure `walk`. Dunque, un file .mat = una registrazione CSI = un campione di una specifica attività. Nel problema di HAR, il modello impara ad associare un segnale CSI ad un'attività svolta, non all'identità della persona che la ha svolta (ID/Re-ID). 

Info sul dataset NTU-Fi HAR:
NTU-Fi HAR — scheda del dataset (per il capitolo 3)
Fonte originale
J. Yang, X. Chen, H. Zou, D. Wang, Q. Xu, L. Xie, "EfficientFi: Toward Large-Scale Lightweight WiFi Sensing via CSI Compression", IEEE Internet of Things Journal, 2022. arXiv: 2204.04138. È il riferimento [36] di SenseFi.
SenseFi (Yang et al., Patterns 2023) usa e ridistribuisce il dataset; cita anche [12] (Wang et al., "CAUTION", IEEE IoT J 2022) per il layout di raccolta.
Conferma indipendente: survey Guarino et al., Computer Communications 249 (2026) 108431, tabella dei dataset (in 01_DOCUMENTI).
Acquisizione (EfficientFi)
Parametro	Valore	Fonte
Hardware	2 router TP-Link N750 (uno TX, uno RX), Atheros CSI Tool	EfficientFi
Banda	5 GHz, canale da 40 MHz	EfficientFi
Sottoportanti	114 per coppia di antenne	EfficientFi, SenseFi
Antenne	3 coppie (1 TX × 3 RX)	EfficientFi; survey Guarino et al.
Frequenza di campionamento	500 Hz	EfficientFi
Durata di un campione	2000 pacchetti / 500 Hz = 4 s (calcolato)	—
Dopo sottocampionamento ::4 (SenseFi)	500 pacchetti a 125 Hz, sempre 4 s	—
Soggetti	20 (13 uomini, 7 donne)	EfficientFi
Ripetizioni	20 per soggetto e per attività → 400 campioni/classe nell'articolo	EfficientFi
Ambiente	laboratorio; Fig. 3 di EfficientFi (distanze indicate 6,5 m e 5 m)	EfficientFi
Preprocessing	nessun denoising; solo ampiezza	EfficientFi
Split	8:2 (non specificato se casuale o per soggetto)	EfficientFi, SenseFi
Le 6 attività (nomi originali)

running (run), walking (walk), falling down (fall), boxing (box), circling arms (circle), cleaning floor (clean = pulire il pavimento).

Punti da segnalare nella relazione
Discrepanza sul numero di campioni: l'articolo parla di 400 campioni/classe; la versione pubblica di SenseFi ne ha 200/classe (936 train + 264 test = 1200). Non è documentato come sia stato scelto il sottoinsieme.
Discrepanza sugli ambienti: SenseFi dice "we collect the data in three different layouts"; EfficientFi descrive un solo laboratorio. Da segnalare come incertezza.
Unità dell'ampiezza non specificata (valori ~3–52, media ~42: compatibili con dB).
Numerazione dei file di test: 16–19 e 160–199 per ogni classe.
Asse dei tempi nei grafici: secondi = pacchetto / 500.


------------------
EfficientFi è il paper che ha raccolto i dati: router, 20 persone, 500 Hz, laboratorio. Il suo obiettivo però non era l'HAR in sé, ma comprimere il CSI per trasmetterlo in modo efficiente. L'HAR era il compito usato per verificare che la compressione non facesse perdere informazione.

SenseFi è un benchmark, cioè un confronto standardizzato. Prende 4 dataset, li prepara tutti allo stesso modo (formato, split train/test) e ci addestra sopra 11 modelli con lo stesso protocollo, per capire quali funzionano meglio nel Wi-Fi sensing. Fornisce anche il codice: la libreria con loader e modelli. Il dataset NTU-Fi HAR è stato raccolto da Yang et al. [EfficientFi] ed è distribuito, nella versione pre-elaborata utilizzata in questo lavoro, all'interno del benchmark SenseFi [SenseFi].
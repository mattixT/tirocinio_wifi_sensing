![[Screenshot 2026-09-17 at 00.01.17.png]]

IL dataset NTU-Fi HAR contiene 6 classi: box, circle, clean, fall, run, walk. Il training set contiene 156 campioni per ogni attività, per un totale di 936 campioni, mentre il test set ne contiene 44 per classe, per un totale di 264 campioni. Il dataset è pertanto bilanciato rispetto alle classi, con complessivamente 200 campioni per attività e 1.200 campioni totali e ogni campione viene fornito al modello nella forma 3 x 114 x 500 (3 antenne x 114 sottocanali x 500 istanti temporali). Ogni file contiene un singolo campione CSI acquisito durante l’esecuzione di una determinata attività. L’etichetta del campione è ricavata dalla cartella in cui si trova il file, ad esempio `fall` oppure `walk`. Dunque, un file .mat = una registrazione CSI = un campione di una specifica attività. Nel problema di HAR, il modello impara ad associare un segnale CSI ad un'attività svolta, non all'identità della persona che la ha svolta (ID/Re-ID). 

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

######## INTRODUZIONE, TEORIA, FATTI INTERESSANTI ###########
Si può definire il Wi-Fi sensing come una computer vision cieca: esso non ha bisogno di telecamere, dispositivi indossabili o altri sistemi di rilevazione, ma solo di dispositivi Wi-Fi che sono ormai presenti all'interno di ogni edificio ed abitazione. A partire dal 2025, dopo la nascita dello standard Wi-Fi 802.11bf, progettato da IEEE 802.11bf Task Group (TGbf), il Wi-Fi sensing è diventato parte del classico servizio
Wi-Fi di cui tutti possiamo usufruire. 

Il rilevamento appena presentato è possibile grazie all'analisi degli stessi segnali Wi-Fi, attraverso Received Signal Strength Indicator (RSSI) o Channel State Information (CSI). Al giorno d'oggi, esiste una netta suddivisione tra metodi di rilevazione basati su RSSI e metodi basati su CSI. RSSI misura il livello di potenza ricevuta dopo eventuali perdite di trasmissione, come quelle causate da oggetti e dalla presenza di persone le influenzano la qualità del segnale. Nonostante l'RSSI sia una misura grezza e non dettagliata del segnale Wi-Fi, esso è stato ampiamente impiegato nel Wi-Fi sensing, ad esempio per ricostruzioni di mappe [W. Xue, Q. Li, X. Hua, K. Yu, W. Qiu, and B. Zhou, “A new algorithm for indoor rssi radio map reconstruction,” IEEE Access, vol. 6, pp.76 118–76 125, 2018.], localizzazione [M. T. Hoang, B. Yuen, X. Dong, T. Lu, R. Westendorp, and K. Reddy,
“Recurrent neural networks for accurate rssi indoor localization,” IEEE
Internet Things J., vol. 6, no. 6, pp. 10 639–10 651, 2019.] [Y. Fu, P. Chen, S. Yang, and J. Tang, “An indoor localization algorithm
based on continuous feature scaling and outlier deleting,” IEEE Internet Things J., vol. 5, no. 2, pp. 1108–1115, 2018], tracciamento [A. Booranawong, N. Jindapetch, and H. Saito, “A system for detection and tracking of human movements using rssi signals,” IEEE Sensors J., vol. 18, no. 6, pp. 2531–2544, 2018] e identificazione di persone [V. Bianchi, P. Ciampolini, and I. De Munari, “Rssi-based indoor localization and identification for zigbee wireless sensor networks in smart homes,” IEEE Trans. Instr. Meas., vol. 68, no. 2, pp. 566–575, 2019.]. [Person Re-Identification Through Wi-Fi Extracted Radio Biometric Signatures]

Per quanto riguarda la misura CSI, essa fornisce informazioni più ricche sulla comunicazione wireless, basate sulle caratteristiche di ampiezza e fase per ciascuna sottoportante nel calane Orthogonal Frequency-Division Multiplexing (OFDM). Per questo motivo, il CSI è al centro della maggior parte degli studi riguardanti il Wi-Fi sensing, anche i più complessi, che hanno come obiettivo Person Identification [Transformer-Based Person Identification via Wi-Fi CSI Amplitude and Phase Perturbations] e Person Re-Identification. [Person Re-Identification Through Wi-Fi Extracted Radio Biometric Signatures].

Il CSI viene misurato calcolando la fase e l'ampiezza per ciascuna sottoportante del segnale trasmesso tra una coppia d'antenne, una che invia i dati e l'altra che riceve i dati. Importante è considerare una coppia ordinata di antenne, data l'esistenza dei sistemi Multiple Input Multiple Output (MIMO), in cui più antenne comunicano contemporaneamente. Pertanto, per ogni coppia di antenne (TX, RX), dove TX è l'antenna trasmittente e RX è l'antenna ricevente, e per ogni sottoportante del canale di comunicazione, c'è una misura del CSI. In particolare, il CSI è costituito:.... QUI CI STAREBBE BENE LA FORMULA DELLA MATRICE CSI E SPIEGAZIONE SU AMPIEZZA E FASE

L'ampiezza e la fase di un segnale wireless indicano rispettivamente: l'intensità del segnale durante la comunicazione (altezza d'onda) e lo spostamento del segnale nel tempo e nello spazio. Le onde radio che si propagano nell'aria, anziché all'interno di cavi, sono infatti soggette all'influenza di vari elementi, come pareti, oggetti e corpi umani. DI conseguenza, il segnale che giunge al router ricevente porta con sé le tracce del percorso seguito e degli ostacoli incontrati. Il CSI racchiude tutte queste informazioni cruciali e può essere sfruttato per rilevare la presenza di persone, movimenti [SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing], oggetti e persino funzioni vitali, come la respirazione [BreathID: Respiration Sensing for Human Identification Using Commodity Wi-Fi], analizzando le alterazioni che certi elementi provocano al segnale trasmesso.

Il lavoro svolto durante questo tirocinio si basa proprio su questi concetti tecnici e, più nello specifico, si posiziona all'interno del rilevamento di attività umane tramite analisi dell'ampiezza di segnali Wi-Fi CSI. 

Il sensing basato su tecnologie senza fili presenta una forte scissione interna dettata dalla presenza o assenza di dispositivi ausiliari. Si fa riferimento ad approcci device-based e ad approcci device-free. I primi sono tutti quelli che prevedono l'utilizzo attivo di un dispositivo che permette di ottenere informazioni riguardo la persona che lo porta con sé, mentre gli approcci device-free si basano esclusivamente sull'effetto che gli improvvisi cambiamenti nell'ambiente causano al segnale, dovuti soprattutto a presenza umana e movimento. Fonte: [A survey on CSI-based Wi-Fi sensing datasets and models with a focus on reproducibility. 2026.]








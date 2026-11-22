# Human activity recognition from Wi-Fi CSI amplitude — NTU-Fi HAR

Code of the internship report *Riconoscimento di attività umane tramite l'ampiezza del CSI Wi-Fi:
riproduzione e analisi del protocollo di valutazione su NTU-Fi HAR* (Human activity recognition from
Wi-Fi CSI amplitude: reproduction and analysis of the evaluation protocol on NTU-Fi HAR), Bachelor's
degree in Computer Science, Sapienza University of Rome; supervisor Prof. Danilo Avola.

It reproduces three models of the [SenseFi](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark)
benchmark (CNN-5, GRU, LSTM) on the NTU-Fi HAR dataset and analyses its evaluation protocol.

## Installation

Python 3.9.6, with the exact versions in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run every script from the root folder of the repository, with the environment activated.
The figures in `work_figures/` also require LaTeX (pdflatex).

## Data and folders

The dataset is not included; download it following the instructions of the SenseFi repository.
The scripts use absolute paths, defined by the constants at the top of each file (`BASE`,
`DATASET`, `EXPERIMENTS`, `FIGURE`); change them if your folder is different:

```
TIROCINIO_TESI/
├── 02_CODICE/tirocinio_wifi_sensing/   this repository
├── 03_DATASET/NTU-Fi_HAR/              train_amp/ and test_amp/, one subfolder per activity
├── 04_ESPERIMENTI/                     results of the runs
├── 05_FIGURE/                          figures of the analysis scripts
└── 07_RELAZIONE/images/                figures of the report
```

## Structure

```
pipeline/              preprocessing, data loading, models, training
results_analysis/      summaries, statistical tests and error analysis (they read 04_ESPERIMENTI)
dataset_exploration/   statistics and figures of the dataset
utilities/             check that the preprocessing matches the SenseFi one
work_figures/          figures of the report (make_all.py regenerates all of them)
```

## Usage

**Training.** Each command trains and tests one model with one seed and saves the weights and
`results.json` in `04_ESPERIMENTI/<model>[_val-<mode>][_norm-<mode>]_seed<N>/`.

```bash
python pipeline/train.py --model <cnn5|gru|lstm> --seed <0-4> [--device <auto|cpu|mps|cuda>]
                         [--val <none|random|block>] [--norm <sensefi|sample|subcarrier>]
```

All the 75 runs of the report (3 models × 5 configurations × 5 seeds):

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

In the report the CNN-5 was trained on the Apple GPU (`--device auto`) and the GRU and LSTM on the
CPU: with the same seed and the same device the results are identical, on different devices they
may differ slightly.

**Analysis of the results**

```bash
python results_analysis/summary.py --model <cnn5|gru|lstm> [--val ...] [--norm ...]   # summary and figures of one model
python results_analysis/stats_report.py                                              # all the numbers of the report
python results_analysis/error_analysis.py --model <cnn5|gru|lstm>                    # misclassified samples
python results_analysis/errors_by_file.py                                            # figure: errors per sample number
python results_analysis/compare_files.py --figure walk                               # figure: misclassified vs correct samples
```

**Dataset and checks**

```bash
python dataset_exploration/dataset_analysis.py           # statistics of the dataset
python utilities/check_preprocessing.py                  # preprocessing identical to SenseFi
python pipeline/evaluate.py --run gru_seed0              # re-check a saved run
```

**Figures of the report**

```bash
python work_figures/make_all.py [--skip-slow]            # avoid to read the entire dataset
```

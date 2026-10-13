\# FedCAV



\*\*Prototype-Guided Federated Continual Video Recognition Under Heterogeneous Client Partitions\*\*



Official research and reproducibility repository for the FedCAV  



\## Overview



FedCAV investigates federated continual video recognition under heterogeneous client data distributions and sequentially introduced action classes.



The experimental framework combines TimeSformer-based video representations, federated model aggregation, prototype-based rehearsal, and replay-weight control.



\## Datasets



\- \*\*UCF101:\*\* 101 action classes, five incremental stages.

\- \*\*HMDB51:\*\* 51 action classes, evaluated under continual-learning settings.



Datasets must be obtained separately from their respective providers.



\## Experimental Configuration



The UCF101 `e4v7\_full` configuration uses:



\- Backbone: TimeSformer

\- Input: 8 video frames at 224 × 224 resolution

\- Federated clients: 2

\- Dirichlet alpha: 0.5

\- Communication rounds per stage: 1

\- Local epochs: 1

\- Prototype memory: 10 per class



Configurations differ across experiments. Consult the saved checkpoint arguments and corresponding scripts for exact settings.



\## Qualitative Video Evaluation



The repository includes `scripts/fedcav\_qualitative\_inference.py`, which evaluates matched adaptive FedCAV and fixed-replay checkpoints on UCF101 videos.



Example:



```powershell

python scripts/fedcav\_qualitative\_inference.py --project D:\\FedCAV --split-root D:\\FedCAV\\data\\ucfTrainTestlist --seed 42 --stage 5

```



Outputs include:



\- `matched\_predictions.csv` — video-level predictions, confidence scores and correctness

\- `decode\_errors.csv` — video decoding failures, if any



The evaluation supports comparisons of global-model recognition outcomes. Individual client predictions require separately retained client-model states.



\## Reproducibility



Training scripts and experiment-specific configurations are organized under `scripts/`. Checkpoints and results are stored in their respective local directories where available.



Large datasets and model checkpoints may not be hosted in this repository.



\## Code Availability



https://github.com/dr-kk/FedCAV



\## Citation



If this repository supports your research, please cite the associated FedCAV manuscript when its bibliographic details become available.



\## Status



Research reproducibility artifacts. Experimental outcomes should be interpreted according to their specific configurations, datasets and evaluation protocols.


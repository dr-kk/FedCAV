E9/E10: derived from the supplied e8_controller_ablation.py.
Run from your FedCAV repository root, with the same dataset/split paths and settings as your original E8 runs.
Example smoke check (NOT publication result):
python scripts/e9_e10_fedcav.py --help
For seed 42 full-run controls, use the commands in RUN_E9_E10.ps1 after editing DATA_ROOT and SPLIT_ROOT.
E9 new modes: fixed_125, fixed_135, fixed_150, stage_ramp; existing full/fixed/drift_only/plasticity_only unchanged.
E10: e10_controller_trace.csv logs previous drift/plasticity, drift scale, transformed signal, preclip/applied lambda per client/stage. Original drift-history CSV remains.
Note: this trace captures controller INPUT at each client-stage, whereas raw drift measured after training is in the original drift-history CSV. Do not confuse them.
Do not infer causality until matched full-run comparisons are complete.

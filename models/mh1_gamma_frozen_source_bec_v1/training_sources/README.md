# Training and fine-tuning source

These files are copied from the original MACELES-MPtrj-uiu research workspace
to preserve the code used to train the selected model chain. `source_manifest.json`
records SHA-256 hashes for every copied Python file.

## Entry points

| File | Role |
| --- | --- |
| `foundation_displacement_train.py` | Trains the frozen-backbone MH-1 Γ projection/head. The selected checkpoint is 512-channel, pre-product, identity-initialized, seed 123, best epoch 205. |
| `foundation_displacement_extract.py` | Extracts the frozen MH-1 features used by Γ training. |
| `foundation_displacement_finetune.py` | Follow-up displacement fine-tuning experiments and checkpoint-loading helpers. It is included for provenance; the selected epoch-205 Γ head is frozen. |
| `finetune_uiu_backbone_displacement.py` | Shared feature/data and backbone fine-tuning helpers imported by the Γ scripts. |
| `finetune_uiu_bec_jarvis.py` | Fine-tunes the MACELES BEC source model on JARVIS BEC labels. |
| `train_uiu_physical_polarization_response.py` | Trains the differentiable-polarization head used by the frozen-source BEC pipeline. |

The remaining `.py` files define the equivariant Γ/BEC response modules,
irreducible tensor conversions, JARVIS/ASE record readers, losses, and local
helpers imported by those entry points. `evaluate_displacement_increment_locked_test.py`
is included because the Γ training modules import its checkpoint/model loader;
it also contains evaluation helpers.

## Inputs and environment

These are source snapshots, not self-contained training jobs. The original
scripts expect a compatible PyTorch, MACE, e3nn, ASE, SciPy, pandas, and
PyTorch Geometric environment, plus research-workspace paths and files such as:

- the upstream MACE-MH-1 checkpoint and its `omat_pbe` head;
- an OMat24 MACELES starting checkpoint for BEC fine-tuning;
- JARVIS DFPT records and the safe5 train/validation splits;
- DFPT response caches, quality summaries, and cached MH-1 features;
- output directories under `results/`.

None of those training datasets or caches is bundled here. Several defaults
refer to the original absolute workspace paths; review and update the paths or
pass the available command-line options before running the scripts.

The physical-polarization trainer supports `--skip-test`. Its historical
training manifest records `skip_test: false`, so the original run also emitted
test-set evaluation metrics after training. The weights in this model bundle
are fixed artifacts; do not use the old test split to choose new checkpoints.

The scripts are included as research source snapshots. This folder does not
add a separate license grant for them; the model bundle's `LICENSE` and all
upstream dependency/model terms still need to be observed.

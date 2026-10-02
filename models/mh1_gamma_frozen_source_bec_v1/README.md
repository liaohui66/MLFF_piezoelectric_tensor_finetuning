# MH-1 Γ + frozen-source BEC (selected model chain)

This bundle records the model selected in `Iteration.md` §63: the frozen
MH-1 pre-product identity-projection displacement head from epoch 205, paired
with the frozen-source BEC prediction chain. The JARVIS-only BEC calibrator is
an evaluated comparator and is not part of this selected chain.

## Files in this bundle

| File | Purpose |
| --- | --- |
| `weights/gamma_head/mh1_gamma_epoch205.pt` | Learned Γ displacement projection and response head (1,209,345 parameters). |
| `weights/bec_source/mace_uiu_jarvis_bec_ft_best.model` | Fine-tuned MACE BEC source model. |
| `weights/bec_source/physical_polarization_head.pt` | Trained differentiable-polarization head used by the frozen-source BEC pipeline. |
| `external/mh1_backbone.json` | External MH-1 model URL, expected SHA256, and upstream license note. |
| `gamma_training_manifest.json` | Original Γ checkpoint architecture and training/validation metadata. |
| `bec_source_training_manifest.json` | BEC source model training metadata. |
| `physical_polarization_training_manifest.json` | Physical-polarization head training metadata. |
| `model_manifest.json` | File hashes, model composition, and selected evaluation metrics. |
| `LICENSE` | Academic Software License terms for this model bundle. |

The MH-1 foundation model itself is intentionally not mirrored here. Run
`./download_mh1_backbone.sh` to fetch the exact upstream artifact after
reviewing its license. The expected SHA256 is checked by the script.

## Prediction chain

1. Load the upstream MACE-MH-1 model with its `omat_pbe` head. For each input
   structure, extract the last-two-interaction `0e`/`1o` node features before
   product blocks, on the unstrained structure and six `±0.005` Voigt-strain
   pairs.
2. Apply the epoch-205 projection/response head to predict the atom-resolved
   internal displacement derivative Γ.
3. Run the frozen MACELES BEC source model and physical-polarization head to
   obtain atom-resolved BECs. Apply the source-preserving identity BEC mapping
   with the acoustic-sum-rule projection. Do not substitute the JARVIS-only
   calibrator for the selected model.
4. Contract BEC with Γ and divide by volume, using the project conversion
   factor `16.0217663 C/m² per e/Å²`.

The final predictor is a two-branch model chain. `mh1_gamma_epoch205.pt` alone
is only the learned Γ head; it does not contain the MH-1 foundation weights.

## Reported evaluation

On the previously used fixed 208-structure safe5 test, MH-1 epoch-205 Γ plus
frozen-source BEC achieved unified ionic component MAE/Pearson
`0.068694 / 0.736669 C/m²` and eijmax MAE/Pearson
`0.216209 / 0.786968 C/m²`. It was the best complete chain among the
pre-specified comparisons in §63. This set is not a new external test set.
Only aggregate metrics are included here; test structures, labels, and
per-structure predictions are not included.

## Reproducibility limits

This release contains the learned checkpoints, their manifests, and the exact
external MH-1 dependency record. It does not yet provide a standalone
one-command inference CLI for arbitrary ASE structures. The original replay
scripts in the research workspace depend on that workspace's datasets,
feature caches, and source modules. A portable inference CLI should be added
before describing this repository as a turnkey predictor.

## License and attribution

The upstream MACE-MH-1 weights are listed by ACEsuit as Academic Software
License (ASL), which limits use to academic non-commercial purposes unless a
separate commercial license is obtained. The combined predictor depends on
that model, so this model bundle is distributed under ASL as well. See
[`LICENSE`](LICENSE), [`external/mh1_backbone.json`](external/mh1_backbone.json),
and the [upstream MACE foundation models](https://github.com/ACEsuit/mace-foundations)
for the current terms and attribution. The upstream weight file is not
redistributed in this repository.

The project's training and inference source code is not included in this
bundle and is not licensed by this model-directory license.

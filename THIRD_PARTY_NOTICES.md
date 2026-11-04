# Third-party sources and attribution

The project's code is distributed under AGPL-3.0-or-later, except where an accompanying upstream notice specifies a compatible separate license. Preserve existing file headers and the accompanying licenses when redistributing.

## THRL

- Repository: https://github.com/touhourl/thrl
- Referenced revision: `41c4f77931a541849a31411bf766f93bbad5e597`.
- The observation/reward/PPO research lineage is recorded in the `thrl_*` modules. Tensor-only PPO arithmetic in `fly_gru_aligned_ppo.py` derives from that lineage.
- Original attribution: T. Liu and contributors; AGPL-3.0-or-later. The full AGPL text is provided in `COPYING` and `LICENSE`; source headers remain in place.
- The released SNN's compact129 observations, learned readouts and subsequent reward revisions are project-specific. This is not a claim that all current behavior duplicates THRL.

## nokozero

- Repository: https://github.com/astral4/nokozero
- Author: Kyle Chen.
- Parts of the standalone fake Direct3D9 backend (`d3d9.rs`, `backing.rs`, `out.rs`) were adapted from nokozero's headless code. See `native/fake_d3d9/NOTICE.md` for the changes and scope.
- Those components retain **MIT OR Apache-2.0**; both complete license texts accompany the Rust source and native binary attachment.
- No nokozero multi-expert neural checkpoint is used by this release.

## Drosophila brain model / FlyWire data

- Preprocessed connectivity source: https://github.com/philshiu/Drosophila_brain_model at `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`.
- Its code license is MIT, copyright Philip Shiu and Nico Spiller; the download script fetches the upstream LICENSE alongside the input data and model.py.
- Annotation source: https://github.com/flyconnectome/flywire_annotations at `ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0` (v2.1.0, materialization 783).
- Annotation sources and exact download records are in `release-data/annotation-sources.json`. The shipped port-order metadata was generated from these sources; its original provenance is retained. The filename's historical 1024-input selection is not the current full-VP selection.
- Data and annotation use follows the respective upstream terms and citation requirements; the project's AGPL license does not replace them. Bulk upstream datasets are downloaded from their original source, not republished in this repository.

Relevant original research:

- Dorkenwald et al., *Neuronal wiring diagram of an adult brain*, Nature 634, 124–138 (2024). https://doi.org/10.1038/s41586-024-07558-y
- Shiu et al., *A Drosophila computational brain model reveals sensorimotor processing*, Nature 634, 210–219 (2024). https://doi.org/10.1038/s41586-024-07763-9
- Schlegel et al., *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature (2024). https://doi.org/10.1038/s41586-024-07686-5
- Lappalainen et al., *Connectome-constrained networks predict neural activity across the fly visual system*, Nature 634, 1132–1140 (2024). https://doi.org/10.1038/s41586-024-07939-3

## Touhou Project

Touhou 14 / Double Dealing Character is a game by Team Shanghai Alice. This repository is an independent research controller, not an official game distribution. No game executable, BGM, graphics or other game assets are included; users must provide their own compatible game installation.

## Other dependencies and historical tools

Python/Rust dependencies retain their own licenses. Historical source modules may mention additional research projects or contain their own attribution headers; those headers are retained. Dependencies are installed normally rather than vendored into this source release.

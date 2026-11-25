# Molecular Substructure Inference from Local Peaks in Tandem Mass Spectra

Given one MS/MS spectrum, this project infers, for every non-precursor peak, a ranked list of
candidate molecular substructures (fragments). Inference uses only the peak list and the acquisition
conditions (precursor m/z, collision energy, ion mode). It does not use a fragmentation DAG or a known
precursor structure.

Training labels are weak: they come from fragmentation DAGs produced by AIMe, which often link one peak
to several possible structures. The output is therefore a ranking of hypotheses, not a verified
chemical identification.

> Research code, work in progress. This repository holds code and technical documentation only.
> Datasets, trained checkpoints and experimental results are not included.

## How it works

1. **Spectrum encoder.** A Transformer embeds each peak in the context of the whole spectrum, with
   Fourier mass features and an attention bias on pairwise mass differences.
2. **Fragment encoder.** A bond-aware message-passing network embeds candidate fragment graphs together
   with their reported ion formula, mass and mode.
3. **Mass-gated retrieval.** Each peak is compared only with library fragments of the same ion mode
   inside its mass tolerance.
4. **Relation supervision.** Four kinds of fragment relations from training DAGs supervise a low-rank
   pairwise scorer.
5. **Joint reranking.** Candidates of different peaks can exchange compatibility messages so that the
   explanations of a spectrum support each other.

See [docs/MODEL.md](docs/MODEL.md) for the full design and its limits.

## Layout

```text
src/casmi_peaks/    Peak-to-fragment data pipeline, chemistry features, base retrieval model
src/casmi_joint/    Joint model: relations, reranking, query/scaffold-disjoint protocol, training, inference
src/casmi_neural/   Earlier DAG-attribution model and shared training utilities
configs/            Pretraining and fine-tuning configurations
tests/              Unit tests on small synthetic molecules
scripts/            Ablation-config generator and synthetic smoke-cache builder
docs/MODEL.md       Technical documentation
```

## Install and test

Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests
```

## Usage

```bash
casmi-joint --help                # training, evaluation and inference commands
casmi-peaks --help                # peak-label preparation and the base model
```

Training needs a fragmentation-DAG archive and a prepared reference cache, which are not distributed here.

## Contact

Yiyu Qian, Cornell University

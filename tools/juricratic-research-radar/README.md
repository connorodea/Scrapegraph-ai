# Juricratic Research Radar

API-first discovery and local indexing of legal machine-learning repositories, papers, datasets, benchmarks, models, annotations, and training pipelines.

The reference pattern is `connorodea/bidaraciv` and the paper *A model for predicting court decisions on child custody*. That project joins a defined jurisdiction and outcome, 3,000 appellate judgments, human-labeled legal grounds, BRAT annotations, model-ready variables, and a linked publication. The Radar generalizes that full research chain across GitHub.

## Capabilities

- Expands a structured legal-research ontology into deterministic GitHub repository searches.
- Recursively shards broad searches by repository creation date and, when necessary, star bands.
- Stores repository metadata, README text, topics, license, file-tree signals, citation files, DOI/arXiv links, dataset/model URLs, and pipeline evidence.
- Scores each result from 0–100 using legal-domain, task, data, method, paper, reproducibility, quality, and penalty components.
- Indexes results in SQLite/FTS5 for fast local research.
- Learns new vocabulary from the indexed corpus using anchor-conditioned co-occurrence.
- Optionally snapshots selected high-scoring repositories as commit-pinned tarballs with SHA-256 and provenance manifests.

## Install

```bash
cd tools/juricratic-research-radar
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export GITHUB_TOKEN="github_pat_..."
```

## First run

```bash
juricratic-radar init
juricratic-radar discover --max-queries 10 --max-repositories 500
juricratic-radar enrich --limit 250
juricratic-radar derive-keywords --minimum-score 25
juricratic-radar search "custody outcome prediction"
juricratic-radar export ./data/legal_research_catalog.jsonl --minimum-score 30
```

Broader recurring loop:

```bash
juricratic-radar run --enrich-limit 1000
juricratic-radar stats
juricratic-radar queries --limit 200
```

Opt-in snapshots:

```bash
juricratic-radar snapshot --minimum-score 70 --maximum-count 20 --maximum-repo-mb 200
```

## Why this does not blindly clone all of GitHub

The default pipeline retains the research metadata and provenance needed to identify and reproduce valuable work. Mirroring every repository would create avoidable storage, licensing, privacy, and freshness problems. The `snapshot` command creates reproducible copies only for reviewed, high-scoring repositories.

## Search dimensions

The bundled ontology covers:

- **Legal source:** judgments, opinions, case law, dockets, pleadings, contracts, statutes, legal grounds, appellate decisions.
- **Target:** judicial decision prediction, case-outcome prediction, sentencing/charge/statute prediction, argument mining, rationale extraction, legal entailment, precedent retrieval, citation prediction, summarization, procedural forecasting, and judicial behavior.
- **Data:** labeled judgment corpora, manual/expert annotation, BRAT/standoff/BIO labels, train/validation/test splits, data cards, and jurisdiction-specific collections.
- **Methods:** transformers, LegalBERT/BERT/RoBERTa/Longformer, recurrent/attention models, GNNs, forests, boosting, SVM, logistic regression, Bayesian models, embeddings, RAG, knowledge graphs, temporal models, and survival analysis.
- **Provenance:** DOI, arXiv, ACL Anthology, `CITATION.cff`, BibTeX, Zenodo, OSF, Figshare, Hugging Face, Papers with Code, model cards, and data cards.
- **Multilingual terms:** ECHR/ECtHR, EU, CAIL/China, U.S., U.K., India, Canada, Australia, Spain, Aragón, `sentencias`, `fundamentos de derecho`, `argumentación jurídica`, `custodia compartida`, `pensión de alimentos`, and `vivienda familiar`.

## Commands

| Command | Purpose |
|---|---|
| `init` | Create SQLite schema and FTS index. |
| `discover` | Search GitHub and store candidates incrementally. |
| `enrich` | Fetch README, topics, license, tree, citations, and score. |
| `run` | Discover, enrich, and derive keywords. |
| `derive-keywords` | Learn expansion terms from the indexed corpus. |
| `search` | Search the local FTS index. |
| `queries` | Preview the exact query plan. |
| `export` | Export CSV or JSONL. |
| `snapshot` | Archive selected commit-pinned repositories. |
| `stats` | Show index counts. |

The exported catalog is designed to feed Juricratic as a research-prior layer. Models and datasets remain bounded by their jurisdiction, time period, sampling, labels, target definition, and evaluation design; they are not treated as legal truth.
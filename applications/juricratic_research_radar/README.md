# Juricratic Research Radar

A visual, API-first research discovery system for finding legal datasets, court-prediction models, transformer pipelines, annotations, benchmarks, game-theoretic litigation research, behavioral law and economics, causal legal analytics, evidence models, and multi-agent simulation priors.

The application is built inside the ScrapeGraphAI repository and uses the open-source `SmartScraperGraph` on already-approved local page content. It integrates optional ScrapingBee and Scrape.do transports for public-page enrichment without using them to bypass authentication, paywalls, private repositories, or explicit access restrictions.

## What it does

### GitHub Repository Radar

- Uses the official GitHub REST API rather than scraping GitHub pages.
- Expands the reference paper and `bidaraciv` repository into a legal-computation query ontology.
- Finds datasets, corpora, annotation projects, notebooks, model code, checkpoints, pipelines, solvers, and evaluation harnesses.
- Reads public READMEs and selected public repository-tree metadata through the API.
- Ranks each repository for Juricratic fit, reproducibility, data value, methodological value, credibility, freshness, and popularity.

### Research Paper Radar

Searches official scholarly APIs across:

- OpenAlex
- Crossref
- Semantic Scholar
- arXiv

The default query map is intentionally broad. It includes legal judgment prediction, court outcomes, argument mining, game-theoretic litigation, bargaining and settlement, signaling and screening, behavioral law and economics, judicial and counsel behavior, litigation economics, causal inference, temporal state models, evidence and credibility, legal network science, reinforcement learning, and multi-agent simulation.

### LSI semantic expansion

The first search pass produces a corpus of repository descriptions, READMEs, paper titles, and abstracts. The system then performs actual Latent Semantic Indexing using TF-IDF and truncated SVD to discover related terms and run a bounded second pass.

### Visual operations

Every run shows:

- Overall and stage-level progress bars
- Plain-English live status messages
- Query and candidate counters
- Access-policy decisions
- Error and blocked-action counters
- Discovery funnel
- Fit-versus-reproducibility quadrant
- Method landscape
- Source treemap
- Juricratic coverage heatmap
- Research relationship graph
- Per-artifact radar profile
- Persistent run history and audit trail

## Reference profile

The initial ontology is reverse-engineered from:

- **Paper:** *A model for predicting court decisions on child custody* (PLOS ONE, 2021)
- **Repository:** `connorodea/bidaraciv`

The reference contributes the core research pattern:

- Manually labeled court rulings
- Factual findings and legal-principle variables
- Outcome and party-request targets
- Temporal train/test validation
- Logistic regression
- Multilayer perceptron and radial-basis neural networks
- Explainable CHAID decision rules
- Argument-mining and automated labeling as a future direction

The Radar generalizes that pattern into Juricratic's larger litigation-intelligence and simulation thesis.

## Safety model

This is deliberately **not** an unbounded scraper or a mirror of all GitHub data.

1. GitHub is API-only. Private repositories are never requested unless a future deployment is deliberately authorized for them.
2. Scholarly discovery uses official metadata APIs first.
3. Local, internal, cloud-metadata, credential-bearing, and non-HTTP URLs are blocked.
4. `robots.txt` is checked before optional public-page enrichment.
5. `401`, `402`, login, subscription, institutional-access, and authorization pages are hard stops.
6. ScrapingBee and Scrape.do are opt-in and only available for explicitly approved public domains.
7. API keys are loaded from environment variables or session-only password fields and are never written to SQLite, exports, or event logs.
8. Responses are size-bounded and non-text payloads are rejected by default.

## Install

From the repository root:

```bash
python -m pip install -e .
python -m pip install -r applications/juricratic_research_radar/requirements.txt
cp applications/juricratic_research_radar/.env.example \
   applications/juricratic_research_radar/.env
streamlit run applications/juricratic_research_radar/app.py
```

Or:

```bash
applications/juricratic_research_radar/run.sh
```

## Configuration

```dotenv
GITHUB_TOKEN=
SCRAPINGBEE_API_KEY=
SCRAPEDO_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=
OPENALEX_MAILTO=research@your-company.com

JURICRATIC_ENABLE_LLM_ENRICHMENT=false
JURICRATIC_LLM_MODEL=ollama/llama3.2
JURICRATIC_LLM_API_KEY=
JURICRATIC_LLM_BASE_URL=

JURICRATIC_RESPECT_ROBOTS=true
JURICRATIC_ALLOW_PROXY_ESCALATION=false
JURICRATIC_ALLOW_FULL_TEXT_FETCH=false
JURICRATIC_ALLOW_JS_RENDERING=false
```

### GitHub token

A read-only fine-grained token is recommended because authenticated GitHub API calls have a larger rate budget. The application reads only public repository metadata in the current implementation.

### ScrapingBee

The key is sent in the recommended bearer authorization header. The application uses low-cost non-JavaScript mode unless JavaScript rendering is explicitly enabled.

### Scrape.do

The key is sent only to the Scrape.do API and is never included in UI logs. Standard public-page mode is used by default; residential/mobile escalation is intentionally not enabled.

### ScrapeGraphAI

Deterministic research extraction always runs for every candidate. Enable ScrapeGraphAI enrichment to produce a stricter structured schema for a user-selected, top-ranked slice only; the UI exposes a hard per-run cap to avoid unbounded LLM cost. The schema covers research question, data, labels, factual/legal/behavioral/strategic variables, methods, validation, metrics, reusable priors, and limitations.

For local inference:

```dotenv
JURICRATIC_ENABLE_LLM_ENRICHMENT=true
JURICRATIC_LLM_MODEL=ollama/llama3.2
```

For an OpenAI-compatible endpoint:

```dotenv
JURICRATIC_ENABLE_LLM_ENRICHMENT=true
JURICRATIC_LLM_MODEL=openai/gpt-4o-mini
JURICRATIC_LLM_API_KEY=...
JURICRATIC_LLM_BASE_URL=https://your-endpoint.example/v1
```

## Autonomous 4-hour + 4-hour research cycle

The durable CLI runner implements the requested two-stage workflow:

1. **Four-hour discovery:** rotates through the ontology in bounded API-first batches, alternating GitHub and scholarly sources, deduplicating, classifying, ranking, and checkpointing after every batch.
2. **Four-hour refinement:** mines the first phase for LSI terms, methods, variables, datasets, validation designs, and weak Juricratic coverage dimensions, then reruns more focused searches and deeper enrichment.

From the repository root:

```bash
python applications/juricratic_research_radar/scripts/run_research_cycle.py
```

The default is exactly four hours of discovery followed by four hours of refinement. The terminal dashboard shows overall and phase progress bars, live plain-English status, the active batch, and counters. Every batch writes an atomic checkpoint to `data/cycles/<cycle-id>/manifest.json`; one failed provider or query does not destroy the cycle.

Useful validation commands:

```bash
# Exercise batching, rotation, adaptive-seed, and checkpoint logic without network calls
python applications/juricratic_research_radar/scripts/run_research_cycle.py \
  --dry-run --max-batches-per-phase 1 --cooldown-seconds 0

# One small live batch per phase after API keys are configured
python applications/juricratic_research_radar/scripts/run_research_cycle.py --smoke

# Papers only, two hours plus two hours
python applications/juricratic_research_radar/scripts/run_research_cycle.py \
  --mode papers --discover-hours 2 --refine-hours 2
```

Launch multi-hour cycles in `tmux`, `screen`, systemd, Docker, or your normal job runner. Do not rely on a browser tab remaining open. The Streamlit **Autonomous 4h + 4h** tab generates commands and visualizes checkpoint manifests.

## Storage

Runs, candidates, scores, access events, and audit messages are stored in SQLite:

```text
applications/juricratic_research_radar/data/juricratic_research_radar.sqlite3
```

The database includes a local FTS5 research index when the Python SQLite build supports it. Search results can be exported as CSV or structured JSON.

## Tests

```bash
cd applications/juricratic_research_radar
PYTHONPATH=. pytest -q tests
```

The tests cover query expansion, access policy, ranking, persistence, and pipeline behavior with mocked clients. No live API keys are required.

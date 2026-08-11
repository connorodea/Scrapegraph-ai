from juricratic_radar.config import load_config
from juricratic_radar.engine import ArtifactExtractor, ResearchScorer, derive_keywords
from juricratic_radar.models import RepositoryRecord
from juricratic_radar.storage import RadarDatabase


def reference_repository() -> RepositoryRecord:
    return RepositoryRecord(
        github_id=1,
        full_name="example/custody-judgment-prediction",
        owner="example",
        name="custody-judgment-prediction",
        html_url="https://github.com/example/custody-judgment-prediction",
        description="Legal judgment prediction from manually annotated appellate judgments",
        stars=42,
        license_spdx="MIT",
        readme_text=(
            "Dataset of court judgments for child custody outcome prediction. "
            "Two jurists performed BRAT manual annotation of the legal grounds. "
            "The repository includes training data, a transformer model, evaluation, "
            "BibTeX, DOI 10.1371/journal.pone.0258993, and reproducible experiments."
        ),
        tree_paths=[
            "CITATION.cff",
            "requirements.txt",
            "Dockerfile",
            "data/judgments.jsonl",
            "src/train.py",
            "tests/test_model.py",
            "notebooks/evaluation.ipynb",
            "LICENSE",
        ],
    )


def test_reference_style_repository_scores_and_indexes(tmp_path) -> None:
    config = load_config()
    repository = reference_repository()
    artifacts = ArtifactExtractor().extract(
        readme_text=repository.readme_text,
        tree_paths=repository.tree_paths,
    )
    score = ResearchScorer(config).score(repository, artifacts=artifacts)

    assert score.total >= 65
    assert "outcome_prediction" in score.classifications
    assert "dataset_or_corpus" in score.classifications
    assert any(a.identifier == "10.1371/journal.pone.0258993" for a in artifacts)

    with RadarDatabase(tmp_path / "radar.sqlite3") as database:
        database.initialize()
        assert database.upsert_repository(repository) is True
        database.update_enrichment(
            repository,
            tree_truncated=False,
            artifacts=artifacts,
            score=score,
        )
        results = database.search("custody judgments")

    assert len(results) == 1
    assert results[0]["full_name"] == repository.full_name


def test_keyword_derivation_promotes_repeated_legal_context() -> None:
    documents = [
        "legal judgment prediction child custody annotated appellate opinions",
        "court decision prediction custody labels legal grounds",
        "legal judgment prediction transformer custody outcome",
        "unrelated image classification tutorial",
    ]
    terms = {
        item.term
        for item in derive_keywords(
            documents,
            ["legal judgment prediction", "court decision prediction"],
            minimum_documents=2,
            limit=50,
        )
    }
    assert "custody" in terms
    assert "unrelated" not in terms

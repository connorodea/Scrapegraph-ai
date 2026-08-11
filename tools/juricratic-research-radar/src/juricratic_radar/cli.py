"""Command-line interface for Juricratic Research Radar."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .config import default_db_path, github_token, load_config
from .engine import build_queries
from .pipeline import RadarPipeline
from .storage import RadarDatabase

app = typer.Typer(no_args_is_help=True, help="Discover and index legal-AI research repositories, papers, datasets, models, and pipelines.")
console = Console()


def _settings(config_path: Path | None, database_path: Path | None):
    return load_config(config_path), (database_path or default_db_path())


def _warn_token() -> None:
    if not github_token():
        console.print("[yellow]GITHUB_TOKEN is not set. Set a read-only token for reliable discovery.[/yellow]")


@app.command("init")
def initialize(database: Annotated[Path | None, typer.Option("--database", "-d")] = None) -> None:
    target = database or default_db_path()
    with RadarDatabase(target) as db:
        db.initialize()
    console.print(f"[green]Initialized[/green] {target}")


@app.command()
def discover(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    database: Annotated[Path | None, typer.Option("--database", "-d")] = None,
    max_queries: Annotated[int | None, typer.Option("--max-queries")] = None,
    max_repositories: Annotated[int | None, typer.Option("--max-repositories")] = None,
    no_sharding: Annotated[bool, typer.Option("--no-sharding")] = False,
) -> None:
    _warn_token()
    settings, path = _settings(config, database)
    with RadarDatabase(path) as db:
        summary = asyncio.run(RadarPipeline(settings, db, token=github_token()).discover(max_queries=max_queries, max_repositories=max_repositories, shard_queries=not no_sharding))
        console.print_json(summary.model_dump_json())


@app.command()
def enrich(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    database: Annotated[Path | None, typer.Option("--database", "-d")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1)] = 100,
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
) -> None:
    _warn_token()
    settings, path = _settings(config, database)
    with RadarDatabase(path) as db:
        summary = asyncio.run(RadarPipeline(settings, db, token=github_token()).enrich(limit=limit, only_unenriched=not refresh))
        console.print_json(summary.model_dump_json())


@app.command()
def run(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    database: Annotated[Path | None, typer.Option("--database", "-d")] = None,
    max_queries: Annotated[int | None, typer.Option("--max-queries")] = None,
    max_repositories: Annotated[int | None, typer.Option("--max-repositories")] = None,
    enrich_limit: Annotated[int, typer.Option("--enrich-limit", min=1)] = 250,
) -> None:
    _warn_token()
    settings, path = _settings(config, database)
    with RadarDatabase(path) as db:
        pipeline = RadarPipeline(settings, db, token=github_token())
        discovery = asyncio.run(pipeline.discover(max_queries=max_queries, max_repositories=max_repositories))
        enrichment = asyncio.run(pipeline.enrich(limit=enrich_limit))
        keywords = pipeline.derive_and_store_keywords(limit=100)
        console.print(json.dumps({"discovery": discovery.model_dump(), "enrichment": enrichment.model_dump(), "derived_keywords": len(keywords), "database": str(path)}, indent=2, default=str))


@app.command("derive-keywords")
def derive_keyword_command(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    database: Annotated[Path | None, typer.Option("--database", "-d")] = None,
    minimum_score: Annotated[float, typer.Option("--minimum-score", min=0, max=100)] = 25,
    minimum_documents: Annotated[int, typer.Option("--minimum-documents", min=1)] = 2,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1)] = 100,
) -> None:
    settings, path = _settings(config, database)
    with RadarDatabase(path) as db:
        keywords = RadarPipeline(settings, db, token=github_token()).derive_and_store_keywords(minimum_score=minimum_score, minimum_documents=minimum_documents, limit=limit)
        table = Table("Term", "Score", "Docs", "Relevant docs")
        for keyword in keywords:
            table.add_row(keyword.term, f"{keyword.score:.3f}", str(keyword.document_frequency), str(keyword.relevant_document_frequency))
        console.print(table)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Local full-text research query")],
    database: Annotated[Path | None, typer.Option("--database", "-d")] = None,
    minimum_score: Annotated[float, typer.Option("--minimum-score", min=0, max=100)] = 0,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1)] = 25,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    with RadarDatabase(database or default_db_path()) as db:
        db.initialize()
        rows = db.search(query, limit=limit, minimum_score=minimum_score)
        if as_json:
            console.print_json(json.dumps(rows, default=str))
            return
        table = Table("Score", "Repository", "Classifications", "Description")
        for row in rows:
            table.add_row(f"{float(row.get('score') or 0):.1f}", str(row["full_name"]), ", ".join(row.get("classifications") or []), str(row.get("description") or "")[:100])
        console.print(table)


@app.command()
def export(
    output: Annotated[Path, typer.Argument(help="CSV or JSONL output path")],
    database: Annotated[Path | None, typer.Option("--database", "-d")] = None,
    minimum_score: Annotated[float, typer.Option("--minimum-score", min=0, max=100)] = 0,
) -> None:
    with RadarDatabase(database or default_db_path()) as db:
        db.initialize()
        result = db.export(output, minimum_score=minimum_score)
    console.print(f"[green]Exported[/green] {result}")


@app.command()
def snapshot(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    database: Annotated[Path | None, typer.Option("--database", "-d")] = None,
    directory: Annotated[Path | None, typer.Option("--directory")] = None,
    minimum_score: Annotated[float | None, typer.Option("--minimum-score", min=0, max=100)] = None,
    maximum_count: Annotated[int | None, typer.Option("--maximum-count", min=1)] = None,
    maximum_repo_mb: Annotated[int | None, typer.Option("--maximum-repo-mb", min=1)] = None,
) -> None:
    _warn_token()
    settings, path = _settings(config, database)
    with RadarDatabase(path) as db:
        manifests = asyncio.run(RadarPipeline(settings, db, token=github_token()).snapshot(directory=directory, minimum_score=minimum_score, maximum_count=maximum_count, maximum_repo_mb=maximum_repo_mb))
        console.print_json(json.dumps(manifests, default=str))


@app.command()
def stats(database: Annotated[Path | None, typer.Option("--database", "-d")] = None) -> None:
    with RadarDatabase(database or default_db_path()) as db:
        db.initialize()
        console.print_json(json.dumps(db.stats(), default=str))


@app.command()
def queries(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    database: Annotated[Path | None, typer.Option("--database", "-d")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1)] = 100,
) -> None:
    settings, path = _settings(config, database)
    with RadarDatabase(path) as db:
        db.initialize()
        dynamic = db.top_derived_keywords(limit=25)
        for index, query in enumerate(build_queries(settings, dynamic, limit=limit), start=1):
            console.print(f"{index:03d}. {query}")

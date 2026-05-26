"""`ocrscout costs` — aggregate per-page cost columns via DuckDB.

Reads ``<output>/data/train-*.parquet`` (local path, ``s3://``, ``gs://``,
or ``hf://``) and prints a grouped summary of pages, GPU-seconds, token
cost (``litellm_cost``), and infrastructure cost
(``gpu_time_cost`` = ``elapsed_seconds × cost_per_hour / 3600``). The
two cost columns are complementary: ``litellm_cost`` reflects what
LiteLLM's pricing DB knows about (commercial APIs, custom
``model_info`` in the proxy config); ``gpu_time_cost`` reflects what
you actually paid for the GPU. Either or both may be zero depending
on how you priced your run.

DuckDB does the heavy lifting because it scans Parquet directly and
handles ``s3://`` / ``hf://`` natively via httpfs; no need to materialise
the full dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

from ocrscout.cli import app
from ocrscout.log import setup_logging

log = logging.getLogger(__name__)

_KNOWN_GROUP_KEYS: tuple[str, ...] = (
    "model", "gpu_type", "provider", "barcode", "output_format",
)


@app.command("costs")
def costs(
    output: str = typer.Option(
        ..., "--output", "-o",
        help="Path to a previous run's output directory (local, s3://, gs://, "
             "hf://). Reads <output>/data/train-*.parquet.",
    ),
    by: str = typer.Option(
        "model",
        "--by",
        help=(
            "Comma-separated GROUP BY columns. Recognised values: "
            + ", ".join(_KNOWN_GROUP_KEYS)
            + ". Defaults to 'model'."
        ),
    ),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Aggregate cost columns across a run's Parquet shards via DuckDB."""
    setup_logging(verbosity=verbose, quiet=quiet)

    group_keys = [k.strip() for k in by.split(",") if k.strip()]
    unknown = [k for k in group_keys if k not in _KNOWN_GROUP_KEYS]
    if unknown:
        raise typer.BadParameter(
            f"unknown group keys {unknown!r}; allowed: {list(_KNOWN_GROUP_KEYS)}"
        )

    glob = _glob_for(output)
    try:
        import duckdb
    except ImportError as e:
        raise typer.BadParameter(
            "duckdb is required for `ocrscout costs`; install it via "
            "`uv pip install duckdb` or `uv pip install ocrscout`"
        ) from e

    con = duckdb.connect(":memory:")
    # ``s3://`` / ``hf://`` need httpfs; DuckDB auto-loads it on first
    # remote read in recent versions, but installing+loading explicitly
    # avoids a "not yet installed" surprise.
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
    except Exception:  # noqa: BLE001
        pass

    sql = _build_query(glob, group_keys)
    log.debug("DuckDB query:\n%s", sql)
    try:
        df = con.execute(sql).fetch_df()
    except Exception as e:  # noqa: BLE001
        raise typer.BadParameter(
            f"DuckDB query failed against {glob!r}: {e}"
        ) from e

    if df.empty:
        rprint(f"[yellow]No rows in {glob}[/yellow]")
        return

    table = Table(title=f"Cost summary ({glob})")
    for col in df.columns:
        table.add_column(col, style="cyan" if col in group_keys else None)
    for row in df.itertuples(index=False):
        table.add_row(*(_fmt(cell) for cell in row))
    rprint(table)


def _glob_for(output: str) -> str:
    """Build the ``data/train-*.parquet`` glob for a local or remote output."""
    if "://" in output:
        return f"{output.rstrip('/')}/data/train-*.parquet"
    return str(Path(output) / "data" / "train-*.parquet")


def _build_query(glob: str, group_keys: list[str]) -> str:
    by_sql = ", ".join(group_keys)
    return f"""
        SELECT
            {by_sql},
            count(*) AS pages,
            sum(coalesce(elapsed_seconds, 0)) AS gpu_seconds,
            avg(elapsed_seconds) AS avg_s_per_page,
            sum(coalesce(input_tokens, 0)) AS input_tokens,
            sum(coalesce(output_tokens, 0)) AS output_tokens,
            sum(coalesce(litellm_cost, 0)) AS token_cost,
            sum(coalesce(gpu_time_cost, 0)) AS infra_cost
        FROM read_parquet('{glob}')
        GROUP BY {by_sql}
        ORDER BY {by_sql}
    """


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if abs(value) >= 1.0:
            return f"{value:,.2f}"
        return f"{value:.4f}"
    return str(value)

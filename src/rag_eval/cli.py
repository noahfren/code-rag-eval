import importlib
from pathlib import Path

import typer
from rich.console import Console

from rag_eval.interfaces import RAGSystem
from rag_eval.reporting import render_json, render_markdown
from rag_eval.runner import BenchmarkRunner, load_dataset

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)
datasets_app = typer.Typer(help="Dataset helpers")
app.add_typer(datasets_app, name="datasets")


def _load_adapter(adapter_spec: str) -> RAGSystem:
    if ":" not in adapter_spec:
        raise typer.BadParameter("Adapter must be in 'module_path:ClassName' format.")
    module_name, class_name = adapter_spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(f"Module not found: {module_name}") from exc
    try:
        adapter_cls: type[RAGSystem] = getattr(module, class_name)
    except AttributeError as exc:
        raise typer.BadParameter(f"Class '{class_name}' not found in '{module_name}'") from exc
    if not issubclass(adapter_cls, RAGSystem):
        raise typer.BadParameter(f"{class_name} must subclass rag_eval.interfaces.RAGSystem")
    return adapter_cls()


@app.command()
def run(  # type: ignore[override]
    dataset: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to dataset YAML"),
    adapter: str = typer.Option(..., help="Adapter spec in 'module:ClassName' form"),
    top_k: int = typer.Option(10, help="Top-K to request from the adapter"),
    overlap_threshold: float = typer.Option(0.5, help="Line overlap threshold for matches"),
    cache_dir: Path = typer.Option(Path(".rag_eval_cache"), help="Where to cache cloned repos"),
    output: Path | None = typer.Option(None, help="Optional path to write report"),
    fmt: str = typer.Option("json", help="Report format: json|md"),
) -> None:
    """Run a benchmark for the given dataset and adapter."""
    report_format = fmt.lower()
    if report_format not in {"json", "md", "markdown"}:
        raise typer.BadParameter("fmt must be one of: json, md, markdown")

    rag_system = _load_adapter(adapter)
    runner = BenchmarkRunner(rag_system, cache_dir=cache_dir)
    report = runner.run(dataset_path=dataset, top_k=top_k, overlap_threshold=overlap_threshold)

    if report_format == "json":
        content = render_json(report)
    else:
        content = render_markdown(report)

    console.print(content)
    if output:
        output.write_text(content)
        console.print(f"[green]Wrote report to {output}")


@datasets_app.command("list")
def list_datasets(
    directory: Path = typer.Option(Path("datasets"), help="Directory containing dataset YAML files")
) -> None:
    """List datasets available locally."""
    if not directory.exists():
        console.print(f"[yellow]Dataset directory not found: {directory}")
        return
    yaml_files = sorted(directory.glob("*.yaml"))
    if not yaml_files:
        console.print(f"[yellow]No datasets found in {directory}")
        return
    console.print("Datasets:")
    for path in yaml_files:
        console.print(f"- {path}")


@datasets_app.command("validate")
def validate_dataset(
    dataset: Path = typer.Option(..., exists=True, dir_okay=False, help="Dataset YAML to validate")
) -> None:
    """Validate a dataset file."""
    ds = load_dataset(dataset)
    console.print(
        f"[green]Dataset '{ds.name}' is valid with {len(ds.queries)} queries (top_k={ds.top_k})."
    )


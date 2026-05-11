from rich.console import Console

console = Console(highlight=False)

def section(title: str) -> None:
    console.print()
    console.rule(title, align="left", style="Dim")

def info(message: str) -> None:
    console.print(f"    {message}", markup=False, highlight=False)

def warn(message: str) -> None:
    console.print(f"    WARNING  {message}", style="Dim", markup=False, highlight=False)

"""Collect generated figure files under one output directory."""

from pathlib import Path


def organize_figures(outputs: Path) -> int:
    """Move Q1-Q4 figure files after the paper has embedded them."""
    figure_dir = outputs / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for question_dir in (outputs / f"q{i}" for i in range(1, 5)):
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
            for source in question_dir.glob(suffix):
                target = figure_dir / source.name
                source.replace(target)
                moved += 1
    return moved


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(f"Organized {organize_figures(root / 'outputs')} figures")

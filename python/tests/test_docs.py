"""Every Python block in the docs runs, from the repository's root as the guides assume."""

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
PAGES = [ROOT / "docs/python.md"]
BLOCKS = [
    pytest.param(code, id=f"{page.name}:{i}")
    for page in PAGES
    for i, code in enumerate(BLOCK.findall(page.read_text()))
]


@pytest.mark.parametrize("code", BLOCKS)
def test_block_runs(code, monkeypatch):
    monkeypatch.chdir(ROOT)
    exec(compile(code, "<docs>", "exec"), {"__name__": "__docs__"})


def test_every_page_has_blocks():
    assert len(BLOCKS) >= 2, "the pattern finds the pages' blocks"
    assert os.path.isfile(PAGES[0])

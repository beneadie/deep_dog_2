"""Verify the minimal example without loading secrets or calling providers."""

from unittest.mock import AsyncMock

import pytest

from deep_research.integration import ResearchResult


@pytest.mark.asyncio
@pytest.mark.parametrize("status,exit_code", [
    ("completed", 0),
    ("partial", 1),
])
async def test_simple_example(monkeypatch, tmp_path, capsys, status, exit_code):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    from scripts import simple_research

    run = AsyncMock(return_value=ResearchResult(status=status, final_report="# Example report"))
    monkeypatch.setattr(simple_research, "run_research", run)
    monkeypatch.chdir(tmp_path)

    assert await simple_research.main() == exit_code
    run.assert_awaited_once_with(simple_research.QUESTION)
    assert "# Example report" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []

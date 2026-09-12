"""Exercise the documented runner without calling providers."""

import sys

import pytest

from deep_research.integration import CredentialCheck, ResearchResult


@pytest.fixture
def quickstart(monkeypatch):
    # Never load a developer's real credentials into the shared test process.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    from scripts import quickstart
    return quickstart


@pytest.mark.asyncio
@pytest.mark.parametrize("status,text,filename,exit_code", [
    ("completed", "# Report", "report.md", 0),
    ("partial", "# Draft", "report.partial.md", 1),
    ("failed", "", None, 1),
])
async def test_report_output(quickstart, monkeypatch, tmp_path, status, text, filename, exit_code):
    output = tmp_path / "report.md"
    monkeypatch.setattr(sys, "argv", ["quickstart.py", "A research question", "--output", str(output)])
    monkeypatch.setattr(quickstart, "validate_credentials", lambda *args: CredentialCheck())

    async def fake_run(prompt, **kwargs):
        assert prompt == "A research question"
        assert kwargs["config"].subagent_max_iterations >= 4
        return ResearchResult(status=status, final_report=text)

    monkeypatch.setattr(quickstart, "run_research", fake_run)
    assert await quickstart.main() == exit_code
    if filename:
        assert (tmp_path / filename).read_text(encoding="utf-8") == text
    if status != "completed":
        assert not output.exists()


@pytest.mark.asyncio
async def test_missing_keys_stop_before_research(quickstart, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["quickstart.py", "A research question"])
    monkeypatch.setattr(quickstart, "validate_credentials", lambda *args: CredentialCheck(
        missing_required=["EXA_API_KEY"]))

    async def unexpected_run(*args, **kwargs):
        pytest.fail("Research must not start with missing credentials")

    monkeypatch.setattr(quickstart, "run_research", unexpected_run)
    with pytest.raises(SystemExit) as exc:
        await quickstart.main()
    assert exc.value.code == 2

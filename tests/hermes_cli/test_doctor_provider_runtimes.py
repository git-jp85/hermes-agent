"""``Model Provider Runtimes`` doctor check: probe fan-out, env-level dedup, reporting."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hermes_cli import doctor_providers as dp


def _unavailable_line(name: str = "solstice", module: str = "httpx") -> str:
    return (
        f"bundled provider plugin {name} unavailable in this runtime: "
        f"declared dependency {module} is not installed"
    )


def _run(monkeypatch, results, labels=None):
    """Run the check with ``_probe`` stubbed to return *results* in order."""
    labels = labels or [(f"runtime {i}", Path(f"/x/{i}/bin/python")) for i in range(len(results))]
    monkeypatch.setattr(dp, "_runtime_interpreters", lambda: labels)
    queue = list(results)
    monkeypatch.setattr(dp, "_probe", lambda label, python, workspace: queue.pop(0))
    return dp._check_provider_runtimes(False)


def test_unavailable_plugins_are_reported_without_being_an_issue(monkeypatch, capsys):
    f = _run(monkeypatch, [{"providers": 49, "unavailable": [_unavailable_line()], "failed": []}])
    out = capsys.readouterr().out
    assert "49 providers" in out
    assert "unavailable here: solstice" in out
    assert "declared dependency httpx is not installed" in out  # the detail line, so the *reason* survives
    assert f.issues == [] and f.manual_issues == []


def test_failed_plugins_become_a_manual_issue(monkeypatch, capsys):
    f = _run(monkeypatch, [{"providers": 50, "unavailable": [], "failed": ["Failed to load bundled provider plugin zz: RuntimeError: boom"]}])
    out = capsys.readouterr().out
    assert "1 failed to load" in out
    assert f.issues == []
    assert len(f.manual_issues) == 1 and f.manual_issues[0].startswith("runtime 0:")


def test_probe_failure_is_warned_per_runtime(monkeypatch, capsys):
    f = _run(monkeypatch, [{"error": "interpreter missing (its pin no longer resolves)"}])
    out = capsys.readouterr().out
    assert "probe failed: interpreter missing" in out
    assert f.manual_issues == []  # a stale pin is reported, but the check itself did not fail


def test_healthy_runtimes_show_their_provider_count(monkeypatch, capsys):
    _run(monkeypatch, [{"providers": 50, "unavailable": [], "failed": []}])
    assert "50 providers" in capsys.readouterr().out


def test_probe_is_skippable_by_env(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_DOCTOR_SKIP_PROVIDER_PROBE", "1")
    called: list = []
    monkeypatch.setattr(dp, "_runtime_interpreters", lambda: [("runtime 0", Path("/x/0/bin/python"))])
    monkeypatch.setattr(dp, "_probe", lambda *a: called.append(a))
    dp._check_provider_runtimes(False)
    assert called == []
    assert "runtime probe skipped" in capsys.readouterr().out


def test_every_runtime_is_probed_even_when_another_fails(monkeypatch):
    seen: list[str] = []

    def probe(label, python, workspace):
        seen.append(label)
        if label == "runtime 0":
            return {"error": "boom"}
        return {"providers": 3, "unavailable": [], "failed": []}

    monkeypatch.setattr(dp, "_runtime_interpreters", lambda: [("runtime 0", Path("/x/0")), ("runtime 1", Path("/x/1"))])
    monkeypatch.setattr(dp, "_probe", probe)
    dp._check_provider_runtimes(False)
    assert seen == ["runtime 0", "runtime 1"]


def test_runtime_interpreters_dedupes_on_environment_not_binary(monkeypatch, tmp_path):
    """Two venvs and a PM generation sharing one interpreter binary must stay distinct.

    Every venv ``bin/python`` is a symlink to the same store interpreter, so keying
    dedup on the resolved binary would hide exactly the difference (fewer providers in
    the minimal PM runtime) this check exists to surface.
    """
    state = tmp_path / "state"
    for venv in ("env-one", "env-two"):
        (state / "environments" / venv / "venv" / "bin").mkdir(parents=True)
        (state / "environments" / venv / "venv" / "bin" / "python").symlink_to(sys.executable)
    (state / "pm-runtime" / "generations" / "gen-one" / "bin").mkdir(parents=True)
    (state / "pm-runtime" / "generations" / "gen-one" / "bin" / "python").symlink_to(sys.executable)

    import pm.environments as environments
    import pm.paths as paths

    monkeypatch.setattr(environments, "install_state_dir", lambda root: state)
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)

    got = dp._runtime_interpreters()
    labels = [label for label, _ in got]
    assert labels[0] == "current interpreter"
    assert sorted(labels[1:]) == ["app venv env-one", "app venv env-two", "pm runtime gen-one"]


def test_runtime_interpreters_survives_missing_pm_layout(monkeypatch, capsys):
    import pm.paths as paths

    def boom(root):
        raise RuntimeError("packaged tree")

    monkeypatch.setattr(paths, "repo_root", boom)
    got = dp._runtime_interpreters()
    assert got and got[0][0] == "current interpreter"
    assert "runtime layout not enumerable" in capsys.readouterr().out


def test_launcher_import_path_is_repo_then_site_packages(monkeypatch):
    """The launcher runs the store Python with repo + venv site-packages on PYTHONPATH."""
    monkeypatch.setattr(dp, "get_project_root", lambda: "/repo")
    monkeypatch.setattr(
        dp.sys,
        "path",
        ["/repo", "/env/lib/python3.14/site-packages", "/base/lib/python3.14", "", "/other/dist-packages", "/env/lib/python3.14/site-packages"],
    )
    parts = dp._launcher_import_path().split(os.pathsep)
    assert parts == ["/repo", "/env/lib/python3.14/site-packages", "/other/dist-packages"]


def test_probe_mirrors_launcher_path_only_for_the_current_interpreter(monkeypatch, tmp_path):
    """A venv that really lacks a dependency must keep reporting it: only sys.executable mirrors the launcher."""
    monkeypatch.setattr(dp, "_launcher_import_path", lambda: "/repo:/env/site-packages")
    seen: dict[str, str] = {}

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def run(argv, **kwargs):
        seen[argv[0]] = kwargs["env"]["PYTHONPATH"]
        return _Completed()

    monkeypatch.setattr(dp.subprocess, "run", run)
    other = tmp_path / "bin" / "python"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("")

    dp._probe("current interpreter", Path(sys.executable), tmp_path)
    dp._probe("pm runtime deadbeef", other, tmp_path)

    assert seen[str(sys.executable)] == "/repo:/env/site-packages"
    assert seen[str(other)] == str(dp.get_project_root())

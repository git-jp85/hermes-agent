"""``hermes doctor`` — model-provider availability, per runtime.

A model-provider plugin may declare an optional distribution (e.g. ``httpx``);
in a *minimal* interpreter such as the PM runtime venv that import is expected
to be absent, so ``providers`` skips the plugin at DEBUG and every short-lived
isolated process stays quiet. The price is invisibility: nothing in ordinary
output says "this runtime has fewer providers". This check buys the visibility
back by probing each interpreter Hermes actually launches and reporting the
provider count (and any plugin that is unavailable) for each one.

Probing runs the target interpreter in a fresh subprocess with a DEBUG handler
attached to the root logger, so it observes exactly the discovery-time records
a real launch would emit — including the plugin-level messages that the parent
process never sees because they happen inside a child.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hermes_cli.config import get_hermes_home, get_project_root
from hermes_cli.doctor_report import check_info, check_ok, check_warn, doctor_check

_PROBE_TIMEOUT_SECONDS = 60

# Runs *inside* the probed interpreter. Reads argv[1] as the JSON output path.
_PROBE_CODE = r"""
import io, json, logging, sys

target = sys.argv[1]
info = {"providers": None, "unavailable": [], "failed": [], "error": None}
stream = io.StringIO()
handler = logging.StreamHandler(stream)
handler.setLevel(logging.DEBUG)
root = logging.getLogger()
root.setLevel(logging.DEBUG)
root.addHandler(handler)
try:
    import providers

    info["providers"] = len(providers.list_providers())
except BaseException as exc:
    info["error"] = "{}: {}".format(type(exc).__name__, exc)
handler.flush()
for line in stream.getvalue().splitlines():
    if "unavailable in this runtime" in line:
        info["unavailable"].append(line)
    elif "Failed to load" in line:
        info["failed"].append(line)
with open(target, "w", encoding="utf-8") as handle:
    json.dump(info, handle)
"""


def _plugin_name(line: str) -> str:
    marker = "provider plugin "
    if marker in line:
        tail = line.split(marker, 1)[1]
        return tail.split(" ", 1)[0].strip(" :,.")
    return line.strip()


def _runtime_interpreters() -> list[tuple[str, Path]]:
    """Interpreters Hermes launches, as ``(label, python)`` in display order.

    The current interpreter comes first, then every installed app-venv
    generation, then the PM runtime generations — the minimal runtime whose
    missing optional dependencies motivated this check. Duplicates collapse on
    the resolved *environment* rather than the interpreter file (see below),
    and an interpreter whose pin no longer resolves is reported as unavailable
    rather than silently dropped.
    """
    candidates: list[tuple[str, Path]] = [("current interpreter", Path(sys.executable))]
    try:
        from pm.environments import install_state_dir, venv_python
        from pm.paths import repo_root

        state = install_state_dir(repo_root())
        for venv in sorted((state / "environments").glob("*/venv")):
            candidates.append((f"app venv {venv.parent.name[:8]}", venv_python(venv)))
        for generation in sorted((state / "pm-runtime" / "generations").glob("*")):
            candidates.append((f"pm runtime {generation.name[:8]}", venv_python(generation)))
    except Exception as exc:  # pm layout unavailable (packaged/nix tree) — probe what we can
        check_info(f"runtime layout not enumerable ({type(exc).__name__}); probing the current interpreter only")
    # Deduplicate on the *environment*, not the interpreter file: every venv's
    # bin/python is a symlink to the same store interpreter, so resolving the
    # binary would collapse distinct environments into one and hide the very
    # difference this check exists to show.
    seen: set[str] = set()
    unique: list[tuple[str, Path]] = []
    for label, python in candidates:
        try:
            key = str(Path(python).parent.parent.resolve())
        except Exception:
            key = str(python)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, python))
    return unique


def _launcher_import_path() -> str:
    """The import path the launcher hands *this* process, for its own probe.

    The runtime is deliberately *not* a plain venv interpreter: it is the store/base
    Python launched with ``PYTHONPATH=<repo><sep><venv>/site-packages``. Probing
    ``sys.executable`` with the repo alone therefore reports plugins whose declared
    dependency is installed for real launches as "unavailable", which would make this
    check cry wolf about the one interpreter users actually run. Reproduce the
    launcher's path instead: this process's own repo and site-packages entries, in
    that order.
    """
    entries = [str(get_project_root())]
    for entry in sys.path:
        if entry and Path(entry).name in ("site-packages", "dist-packages"):
            entries.append(entry)
    return os.pathsep.join(dict.fromkeys(entries))


def _probe(label: str, python: Path, workspace: Path) -> dict:
    """Import ``providers`` in *python* and report what that interpreter loaded."""
    stem = "".join(character if character.isalnum() else "-" for character in label).strip("-")
    target = workspace / f"providers-{stem}.json"
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", str(Path.home())),
        # Only the interpreter this very process runs on mirrors the launcher; every
        # other runtime is deliberately probed bare, so that a venv which genuinely
        # lacks an optional dependency still shows up as such.
        "PYTHONPATH": _launcher_import_path() if Path(python) == Path(sys.executable) else str(get_project_root()),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if not Path(python).exists():
        return {"error": "interpreter missing (its pin no longer resolves)"}
    try:
        completed = subprocess.run(
            [str(python), "-c", _PROBE_CODE, str(target)],
            cwd=str(get_project_root()),
            env=environment,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if not target.exists():
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        return {"error": detail[-1] if detail else f"probe produced no result (exit {completed.returncode})"}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@doctor_check("Model provider runtime probe failed", "({e})")
def _check_provider_runtimes(should_fix: bool, f) -> None:
    """Report provider count and unavailable plugins for every Hermes runtime."""
    if os.environ.get("HERMES_DOCTOR_SKIP_PROVIDER_PROBE"):
        check_info("runtime probe skipped (HERMES_DOCTOR_SKIP_PROVIDER_PROBE is set)")
        return
    runtimes = _runtime_interpreters()
    if not runtimes:
        check_info("no interpreter found to probe")
        return
    workspace = Path(os.environ.get("TMPDIR", "/tmp")) / "hermes-doctor-providers"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
    except OSError:
        workspace = Path(get_hermes_home())
    with ThreadPoolExecutor(max_workers=min(4, len(runtimes))) as pool:
        results = list(pool.map(lambda item: _probe(item[0], item[1], workspace), runtimes))
    for (label, _python), info in zip(runtimes, results):
        error = info.get("error")
        if error:
            check_warn(f"{label}", f"(probe failed: {error})")
            continue
        count = info.get("providers")
        unavailable = info.get("unavailable") or []
        failed = info.get("failed") or []
        if unavailable:
            names = ", ".join(sorted({_plugin_name(line) for line in unavailable}))
            check_ok(f"{label}", f"({count} providers; unavailable here: {names})")
            for line in unavailable:
                check_info(line)
        elif failed:
            check_warn(f"{label}", f"({count} providers; {len(failed)} failed to load)")
            for line in failed:
                check_info(line)
            f.manual_issues.append(
                f"{label}: {len(failed)} model-provider plugin(s) failed to load — see the rows above"
            )
        else:
            check_ok(f"{label}", f"({count} providers)")

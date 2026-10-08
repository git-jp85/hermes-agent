"""One stdlib JSON-line PM request per isolated process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import sys
import threading


def _read_controls(messages, pause, fd):
    # Raw reads avoid a daemon thread holding sys.stdin's buffered lock at exit.
    pending = b""
    request_id = None
    try:
        while block := os.read(fd, 65536):
            pending += block
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                message = json.loads(line)
                if request_id is None:
                    request_id = message["id"]
                if message["id"] != request_id:
                    raise ValueError("unexpected PM control request id")
                if message.get("cancel"):
                    pause.set()
                if message.get("type") != "cancel":
                    messages.put(message)
    except (OSError, ValueError, KeyError) as exc:
        messages.put(exc)
    finally:
        os.close(fd)
        pause.set()
        messages.put(None)


class _PrefixedStderr:
    """stderr proxy that tags each line with the worker's label.

    ``main()`` points fd 1 at stderr so the PM protocol keeps the pipe; without a
    tag a worker's own log lines (provider discovery, subprocess chatter) land in
    the parent's log stream indistinguishable from the parent's own. Callers that
    know the job set ``HERMES_WORKER_LABEL``; unlabelled workers keep the raw
    stream. Bytes written by C extensions or the interpreter itself bypass this
    proxy — it only makes Python-level writes attributable.
    """

    def __init__(self, stream, prefix: str) -> None:
        self._stream = stream
        self._prefix = prefix
        self._pending = True

    def write(self, text: str) -> int:
        try:
            if not text:
                return 0
            prefix = self._prefix
            pieces = []
            for index, chunk in enumerate(text.split("\n")):
                if index:
                    pieces.append("\n")
                    self._pending = True
                if not chunk:
                    continue
                if self._pending:
                    pieces.append(prefix)
                    self._pending = False
                pieces.append(chunk)
            self._stream.write("".join(pieces))
            return len(text)
        except Exception:
            return len(text)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def main():
    import truststore

    truststore.inject_into_ssl()
    # Capture the protocol FD before redirecting even native/subprocess stdout.
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    label = os.environ.get("HERMES_WORKER_LABEL", "").strip()
    if label:
        sys.stderr = _PrefixedStderr(sys.stderr, f"[{label}] ")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # A pending read on inherited control stdin can block child Python startup
    # on Windows. Keep the protocol private and give every ordinary child EOF.
    controls = os.dup(0)
    os.set_inheritable(controls, False)
    with open(os.devnull, "rb") as null:
        os.dup2(null.fileno(), 0)
    messages = queue.Queue()
    pause = threading.Event()
    threading.Thread(target=_read_controls, args=(messages, pause, controls), daemon=True).start()

    def receive():
        message = messages.get()
        if message is None:
            raise RuntimeError("PM client disconnected")
        if isinstance(message, Exception):
            raise message
        return message

    request = receive()
    from pm import paths, plugin_inputs, receipt
    from pm.package import InstallError
    from pm.registry import load_package_definitions
    from pm.runtime import lease_current_runtime
    from pm.worker_operations import OPERATIONS
    lease_current_runtime()
    context = request["context"]
    paths.repo_root = lambda: Path(context["repo"])
    paths.lockfile_path = lambda: Path(context["lockfile"])

    call = 0
    callback_lock = threading.Lock()

    def send(data):
        wire.write(json.dumps({"id": request["id"], **data}) + "\n")

    def callback(name, *args):
        with callback_lock:
            return exchange(name, args)

    def exchange(name, args):
        nonlocal call
        call += 1
        send({"type": "callback", "callback": name, "call": call, "args": args})
        reply = receive()
        if reply["id"] != request["id"] or reply["call"] != call:
            raise RuntimeError("unexpected PM callback response")
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["result"]

    with receipt.worker_context(request.get("update_id")):
        try:
            load_package_definitions(request.get("packages", []))
            operation = request["operation"]
            implementation = OPERATIONS[operation].resolve(operation)
            arguments = request["arguments"]
            if request["operation"] in ("sync_venv", "venv_is_current"):
                arguments["plugins"] = plugin_inputs.decode(arguments.get("plugins"))
            if request["operation"] == "ensure":
                arguments["pause_event"] = pause
            for name in ("progress", "download_progress"):
                if name in request["callbacks"]:
                    arguments[name] = lambda *args, name=name: callback(name, *args)
            result = implementation(**arguments)
            if request["operation"] == "ensure":
                result = None  # Runner is reconstructed from the caller's base env.
            if isinstance(result, Path):
                result = str(result)
            response = {"result": result}
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, InstallError):
                error.update(package=exc.package, cause=exc.cause, remedy=exc.remedy)
            response = {"error": error}
        response["receipt"] = receipt.last_completed()
    send({"type": "result", **response})


if __name__ == "__main__":
    main()

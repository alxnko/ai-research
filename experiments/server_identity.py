"""Identity of the model the *server* is serving (run-time only, stdlib only).

The launcher already compares the model store on disk with the manifest. That is
the local store, not necessarily what the Ollama process answering requests has
loaded. This module asks the server itself.

Evidence, in order:

 1. `GET /api/tags` before inference: exactly one entry for the requested tag,
    whose `digest` is the registered manifest digest.
 2. `GET /api/ps` after every model call (so after the model is loaded, and
    again after every retry): exactly one loaded entry for the tag, same digest,
    and a `context_length` equal to the context the experiment requested.
 3. `GET /api/tags` and `/api/ps` once more when the engine returns; the digest
    must be the same in every observation. A model re-pulled or swapped while
    the attempt ran therefore fails the attempt.

Fail closed: an unreachable endpoint, malformed JSON, a missing or duplicated
entry, a missing or non-matching digest, or a missing context length is a
problem. Nothing falls back to the local fingerprint.

Same instance: the engine's HTTP client targets a fixed address (the default of
`chat_stream`'s `url`). `instance_problems` reads that default from the loaded
engine and requires it to equal the address inspected here, so the server that
is inspected is the one that receives inference requests. `OLLAMA_HOST` only
configures the server process and the `ollama` CLI; the engine ignores it.

What this is not: attestation. Ollama reports a model *manifest* digest and what
it has loaded; it does not prove which weights produced each token. This is
consistency evidence for the advertised identity, not proof of the bytes used
for any individual request.
"""

from __future__ import annotations

import inspect
import json
import re
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit

BASE = "http://127.0.0.1:11434"
TIMEOUT_S = 5.0
ATTESTATION = "server-advertised (not cryptographic attestation of weights)"
_DIGEST = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")


def normalise_digest(value: Any) -> str | None:
    """`sha256:<hex>` for a full digest in either the prefixed (manifest) or bare
    (server API) form; None for anything else, including the 12-char list id."""
    if not isinstance(value, str):
        return None
    m = _DIGEST.match(value.strip().lower())
    return f"sha256:{m.group(1)}" if m else None


def get_json(path: str, base: str | None = None) -> tuple[dict[str, Any] | None, str]:
    url = (base or BASE) + path
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
            doc = json.load(resp)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        if isinstance(exc, urllib.error.HTTPError):
            exc.close()
        return None, f"{path} unavailable ({type(exc).__name__})"
    if not isinstance(doc, dict):
        return None, f"{path} did not return a JSON object"
    return doc, ""


def _named(doc: dict[str, Any], tag: str) -> tuple[list[dict[str, Any]], str]:
    models = doc.get("models")
    if not isinstance(models, list) or not all(isinstance(m, dict) for m in models):
        return [], "models list missing or malformed"
    want = tag if ":" in tag else f"{tag}:latest"
    return [m for m in models if want in (m.get("name"), m.get("model"))], ""


def _one(
    doc: dict[str, Any], tag: str, want_digest: str, where: str
) -> tuple[dict[str, Any] | None, list[str]]:
    found, why = _named(doc, tag)
    if why:
        return None, [f"{where}: {why}"]
    if not found:
        return None, [f"{where}: {tag!r} is not listed"]
    if len(found) > 1:
        return None, [f"{where}: {tag!r} is listed {len(found)} times (ambiguous)"]
    entry = found[0]
    got = normalise_digest(entry.get("digest"))
    if got is None:
        return None, [f"{where}: no usable digest for {tag!r}"]
    if got != normalise_digest(want_digest):
        return None, [f"{where}: {tag!r} has digest {got}, registered {want_digest}"]
    return entry, []


def advertised(
    tag: str, want_digest: str, base: str | None = None
) -> tuple[list[str], dict[str, Any]]:
    """`/api/tags` (before inference and at completion)."""
    doc, err = get_json("/api/tags", base)
    if doc is None:
        return [err], {}
    entry, problems = _one(doc, tag, want_digest, "/api/tags")
    if entry is None:
        return problems, {}
    return [], {"digest": normalise_digest(entry["digest"])}


def loaded(
    tag: str, want_digest: str, ctx: int, base: str | None = None
) -> tuple[list[str], dict[str, Any]]:
    """`/api/ps`: the model the server has loaded right now."""
    doc, err = get_json("/api/ps", base)
    if doc is None:
        return [err], {}
    entry, problems = _one(doc, tag, want_digest, "/api/ps")
    if entry is None:
        return problems, {}
    seen = entry.get("context_length")
    if seen is None:
        return ["/api/ps: no context_length reported"], {}
    if seen != ctx:
        return [f"/api/ps: loaded context {seen}, the experiment requests {ctx}"], {}
    return [], {
        "digest": normalise_digest(entry["digest"]),
        "context_length": seen,
        "size_vram": entry.get("size_vram"),
        "size": entry.get("size"),
    }


def instance_problems(engine: Any, base: str | None = None) -> list[str]:
    """The inspected server is the one the engine sends requests to."""
    inspected = base or BASE
    try:
        default = inspect.signature(engine.chat_stream).parameters["url"].default
        parts = urlsplit(default)
        target = f"{parts.scheme}://{parts.netloc}"
    except AttributeError, KeyError, TypeError, ValueError:
        return ["cannot determine which server the engine sends requests to"]
    if target != inspected:
        return [
            f"the engine calls {target} but the identity check inspects {inspected}"
        ]
    return []


class Watcher:
    """Collects server-side evidence around one attempt."""

    def __init__(self, tag: str, digest: str, ctx: int, base: str | None = None):
        self.tag, self.digest, self.ctx, self.base = tag, digest, ctx, base
        self.start: dict[str, Any] = {}
        self.observations: list[dict[str, Any]] = []
        self.problems: list[str] = []
        self.end: dict[str, Any] = {}

    def begin(self) -> list[str]:
        problems, self.start = advertised(self.tag, self.digest, self.base)
        return [f"before inference: {p}" for p in problems]

    def observe(self) -> None:
        """After each model call (also after a failed or retried one)."""
        problems, seen = loaded(self.tag, self.digest, self.ctx, self.base)
        self.observations.append(seen)
        n = len(self.observations)
        self.problems += [f"after model call {n}: {p}" for p in problems]

    def finish(self) -> list[str]:
        problems, self.end = advertised(self.tag, self.digest, self.base)
        problems += loaded(self.tag, self.digest, self.ctx, self.base)[0]
        if not self.observations:
            problems.append("the loaded model was never observed")
        return [*self.problems, *(f"at completion: {p}" for p in problems)]

    def record(self) -> dict[str, Any]:
        last = self.observations[-1] if self.observations else {}
        return {
            "instance": self.base or BASE,
            "attestation": ATTESTATION,
            "advertised_digest": self.start.get("digest"),
            "loaded_digest": last.get("digest"),
            "loaded_context": last.get("context_length"),
            "loaded_size_vram": last.get("size_vram"),
            "loaded_checks": len(self.observations),
        }

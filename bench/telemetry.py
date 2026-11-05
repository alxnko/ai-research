"""Inference telemetry, streaming transport, and GPU observability."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

try:
    import importlib

    pynvml: Any = importlib.import_module("pynvml")
    pynvml.nvmlInit()
    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False
    pynvml = None

SAMPLING_VERSION = "v1"

BENCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a shell command inside the task workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash shell command to run",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_task",
            "description": "Signal task completion ready for scoring",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


@dataclass
class SamplingConfig:
    profile: str = "deterministic-v1"
    version: str = SAMPLING_VERSION
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplingConfig:
        return cls(
            profile=data.get("profile", "deterministic-v1"),
            version=data.get("version", SAMPLING_VERSION),
            temperature=float(data.get("temperature", 0.0)),
            top_p=float(data.get("top_p", 0.95)),
            top_k=int(data.get("top_k", 40)),
            repeat_penalty=float(data.get("repeat_penalty", 1.1)),
            seed=int(data.get("seed", 42)),
        )


@dataclass
class StreamResult:
    content: str
    thinking: str
    tool_calls: list[dict[str, Any]]
    telemetry: dict[str, Any]


def parse_stream_chunk_event(
    chunk: dict[str, Any],
    t_start: float,
    state: dict[str, Any],
) -> None:
    """Parse a single streaming JSON chunk from Ollama and update accumulators."""
    now = time.perf_counter()
    msg = chunk.get("message", {})

    content_piece = msg.get("content", "") or ""
    thinking_piece = msg.get("thinking", "") or ""
    tool_calls_piece = msg.get("tool_calls", []) or []

    if (content_piece or thinking_piece or tool_calls_piece) and state.get(
        "ttft_any_s"
    ) is None:
        state["ttft_any_s"] = round(now - t_start, 4)

    if thinking_piece:
        state["thinking"] += thinking_piece
        if state.get("ttft_thinking_s") is None:
            state["ttft_thinking_s"] = round(now - t_start, 4)

    if content_piece:
        state["content"] += content_piece
        if state.get("ttft_content_s") is None:
            state["ttft_content_s"] = round(now - t_start, 4)

    if tool_calls_piece:
        state["tool_calls"].extend(tool_calls_piece)
        if state.get("ttft_tool_s") is None:
            state["ttft_tool_s"] = round(now - t_start, 4)

    if chunk.get("done"):
        state["done"] = True
        state["done_reason"] = chunk.get("done_reason", "stop")
        state["prompt_eval_count"] = int(chunk.get("prompt_eval_count", 0))
        if "prompt_eval_cached_count" in chunk:
            state["prompt_eval_cached_count"] = int(chunk["prompt_eval_cached_count"])
        else:
            state["prompt_eval_cached_count"] = None
        state["prompt_eval_duration_ns"] = int(chunk.get("prompt_eval_duration", 0))
        state["eval_count"] = int(chunk.get("eval_count", 0))
        state["eval_duration_ns"] = int(chunk.get("eval_duration", 0))
        state["total_duration_ns"] = int(chunk.get("total_duration", 0))
        state["load_duration_ns"] = int(chunk.get("load_duration", 0))


def build_telemetry_record(
    state: dict[str, Any],
    wall_duration_s: float,
    step: int = 0,
    request_idx: int = 1,
) -> dict[str, Any]:
    """Construct structured inference telemetry from stream accumulation."""
    eval_dur_s = state.get("eval_duration_ns", 0) / 1e9
    prompt_dur_s = state.get("prompt_eval_duration_ns", 0) / 1e9
    eval_count = state.get("eval_count", 0)
    prompt_count = state.get("prompt_eval_count", 0)
    done = state.get("done", False)
    if not done:
        done_reason = "stream_truncated_without_done"
    else:
        done_reason = state.get("done_reason") or "stop"

    decode_tps = round(eval_count / eval_dur_s, 2) if eval_dur_s > 0 else 0.0
    prompt_tps = round(prompt_count / prompt_dur_s, 2) if prompt_dur_s > 0 else 0.0

    return {
        "step": step,
        "request_idx": request_idx,
        "ttft_s": state.get("ttft_any_s"),
        "ttft_thinking_s": state.get("ttft_thinking_s"),
        "ttft_content_s": state.get("ttft_content_s"),
        "ttft_tool_s": state.get("ttft_tool_s"),
        "prompt_eval_count": prompt_count,
        "prompt_eval_cached_count": state.get("prompt_eval_cached_count", 0),
        "prompt_eval_duration_s": round(prompt_dur_s, 4),
        "prompt_eval_tokens_per_s": prompt_tps,
        "eval_count": eval_count,
        "eval_duration_s": round(eval_dur_s, 4),
        "decode_tokens_per_s": decode_tps,
        "total_duration_s": round(state.get("total_duration_ns", 0) / 1e9, 4),
        "load_duration_s": round(state.get("load_duration_ns", 0) / 1e9, 4),
        "transport_wall_s": round(wall_duration_s, 4),
        "incomplete_stream": not done,
        "done_reason": done_reason,
        "content_length_chars": len(state.get("content", "")),
        "thinking_length_chars": len(state.get("thinking", "")),
        "tool_call_count": len(state.get("tool_calls", [])),
    }


def chat_stream(
    model: str,
    msgs: list[dict[str, Any]],
    ctx: int,
    ngpu: int,
    sampling: SamplingConfig | None = None,
    tools: list[dict[str, Any]] | None = None,
    timeout: int = 1800,
    url: str = "http://127.0.0.1:11434/api/chat",
    step: int = 0,
    request_idx: int = 1,
) -> StreamResult:
    """Stream from Ollama /api/chat with per-request TTFT and token accounting."""
    if sampling is None:
        sampling = SamplingConfig()

    options = {
        "num_ctx": ctx,
        "num_gpu": ngpu,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "repeat_penalty": sampling.repeat_penalty,
        "seed": sampling.seed,
    }

    body: dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "stream": True,
        "options": options,
    }
    if tools:
        body["tools"] = tools

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    state: dict[str, Any] = {
        "content": "",
        "thinking": "",
        "tool_calls": [],
        "done": False,
        "ttft_any_s": None,
        "ttft_thinking_s": None,
        "ttft_content_s": None,
        "ttft_tool_s": None,
        "done_reason": None,
        "prompt_eval_count": 0,
        "prompt_eval_cached_count": 0,
        "prompt_eval_duration_ns": 0,
        "eval_count": 0,
        "eval_duration_ns": 0,
        "total_duration_ns": 0,
        "load_duration_ns": 0,
    }

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            line_str = line.decode("utf-8", "replace").strip()
            if not line_str:
                continue
            chunk = json.loads(line_str)
            parse_stream_chunk_event(chunk, t0, state)

    t_end = time.perf_counter()
    telemetry = build_telemetry_record(
        state,
        wall_duration_s=t_end - t0,
        step=step,
        request_idx=request_idx,
    )

    return StreamResult(
        content=state["content"],
        thinking=state["thinking"],
        tool_calls=state["tool_calls"],
        telemetry=telemetry,
    )


def summarize_inference(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-request telemetry into task-level inference summary."""
    if not records:
        return {
            "total_requests": 0,
            "total_prompt_tokens": 0,
            "total_eval_tokens": 0,
            "total_tokens": 0,
            "mean_ttft_s": None,
            "mean_decode_tokens_per_s": 0.0,
            "mean_prompt_eval_tokens_per_s": 0.0,
            "total_eval_duration_s": 0.0,
            "total_prompt_eval_duration_s": 0.0,
            "done_reasons": [],
            "max_prompt_tokens": 0,
        }

    total_prompt = sum(r.get("prompt_eval_count", 0) for r in records)
    total_eval = sum(r.get("eval_count", 0) for r in records)
    total_prompt_dur = sum(r.get("prompt_eval_duration_s", 0.0) for r in records)
    total_eval_dur = sum(r.get("eval_duration_s", 0.0) for r in records)

    ttfts = [r["ttft_s"] for r in records if r.get("ttft_s") is not None]
    mean_ttft = round(sum(ttfts) / len(ttfts), 4) if ttfts else None

    dec_speeds = [
        r["decode_tokens_per_s"]
        for r in records
        if r.get("decode_tokens_per_s", 0.0) > 0
    ]
    mean_decode_tps = round(sum(dec_speeds) / len(dec_speeds), 2) if dec_speeds else 0.0

    prompt_speeds = [
        r["prompt_eval_tokens_per_s"]
        for r in records
        if r.get("prompt_eval_tokens_per_s", 0.0) > 0
    ]
    mean_prompt_tps = (
        round(sum(prompt_speeds) / len(prompt_speeds), 2) if prompt_speeds else 0.0
    )

    done_reasons = [r.get("done_reason", "unknown") for r in records]
    max_prompt = max((r.get("prompt_eval_count", 0) for r in records), default=0)

    token_weighted_decode_tps = (
        round(total_eval / total_eval_dur, 2) if total_eval_dur > 0 else 0.0
    )
    token_weighted_prompt_tps = (
        round(total_prompt / total_prompt_dur, 2) if total_prompt_dur > 0 else 0.0
    )

    return {
        "total_requests": len(records),
        "total_prompt_tokens": total_prompt,
        "total_eval_tokens": total_eval,
        "total_tokens": total_prompt + total_eval,
        "mean_ttft_s": mean_ttft,
        "mean_decode_tokens_per_s": mean_decode_tps,
        "mean_prompt_eval_tokens_per_s": mean_prompt_tps,
        "token_weighted_decode_tokens_per_s": token_weighted_decode_tps,
        "token_weighted_prompt_eval_tokens_per_s": token_weighted_prompt_tps,
        "total_eval_duration_s": round(total_eval_dur, 4),
        "total_prompt_eval_duration_s": round(total_prompt_dur, 4),
        "done_reasons": done_reasons,
        "max_prompt_tokens": max_prompt,
    }


def check_gpu_telemetry() -> str:
    """Detect whether GPU telemetry via pynvml or nvidia-smi is available."""
    if HAS_PYNVML:
        return "available"
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            timeout=3,
        )
        if res.returncode == 0:
            return "available"
        return "unsupported"
    except FileNotFoundError, OSError, subprocess.SubprocessError:
        return "unavailable"


def _query_pynvml_memory() -> dict[str, Any] | None:
    """Attempt fast in-process query via NVML."""
    if not HAS_PYNVML or pynvml is None:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mib = int(mem_info.total // (1024 * 1024))
        used_mib = int(mem_info.used // (1024 * 1024))
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        total_ollama = 0
        for p in procs:
            try:
                name = (
                    pynvml.nvmlSystemGetProcessName(p.pid)
                    .decode("utf-8", "replace")
                    .lower()
                )
            except Exception:
                name = ""
            if "ollama" in name or "llama-server" in name:
                total_ollama += int(p.usedGpuMemory // (1024 * 1024))
        ollama_mib = total_ollama if total_ollama > 0 else None
        return {
            "device_vram_total_mib": total_mib,
            "device_vram_used_mib": used_mib,
            "ollama_vram_used_mib": ollama_mib,
            "telemetry_status": "available",
            "telemetry_source": "pynvml",
        }
    except Exception:
        return None


def _query_nvidia_smi_memory() -> tuple[int | None, int | None, int | None]:
    """Fallback query via nvidia-smi CLI."""
    total_mib: int | None = None
    used_mib: int | None = None
    ollama_mib: int | None = None

    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            timeout=3,
        )
        if res.returncode == 0:
            parts = [p.strip() for p in res.stdout.decode().split(",") if p.strip()]
            if len(parts) >= 2:
                total_mib = int(parts[0])
                used_mib = int(parts[1])
    except Exception:
        pass

    try:
        apps_res = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            timeout=3,
        )
        if apps_res.returncode == 0:
            total_ollama = 0
            for raw_line in apps_res.stdout.decode().splitlines():
                parts = [p.strip() for p in raw_line.strip().split(",") if p.strip()]
                if len(parts) >= 3:
                    pname = parts[1].lower()
                    if "ollama" in pname or "llama-server" in pname:
                        total_ollama += int(parts[2])
            if total_ollama > 0:
                ollama_mib = total_ollama
    except Exception:
        pass

    return total_mib, used_mib, ollama_mib


def get_gpu_memory() -> dict[str, Any]:
    """Query total device VRAM, used device VRAM, and Ollama-specific VRAM."""
    status = check_gpu_telemetry()
    if status != "available":
        return {
            "device_vram_total_mib": None,
            "device_vram_used_mib": None,
            "ollama_vram_used_mib": None,
            "telemetry_status": status,
            "telemetry_source": "none",
        }

    pynvml_res = _query_pynvml_memory()
    if pynvml_res is not None:
        return pynvml_res

    tot, used, ollama = _query_nvidia_smi_memory()
    return {
        "device_vram_total_mib": tot,
        "device_vram_used_mib": used,
        "ollama_vram_used_mib": ollama,
        "telemetry_status": "available",
        "telemetry_source": "nvidia-smi",
    }


def get_ollama_offload_status(
    model_name: str, url: str = "http://127.0.0.1:11434/api/ps"
) -> dict[str, Any]:
    """Query Ollama /api/ps to inspect model memory residency and runner details.

    Note: reported_vram_size_ratio compares size_vram against model size; it is a
    memory byte fraction reported by the runner, not a count of resident layers.
    """
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.load(resp)
        models = data.get("models", [])
        for m in models:
            m_name = m.get("name", "")
            m_model = m.get("model", "")
            matches = (
                model_name in (m_name, m_model)
                or model_name in m_name
                or m_name in model_name
            )
            if matches:
                size = m.get("size", 0)
                size_vram = m.get("size_vram", 0)
                ratio = round(size_vram / size, 4) if size > 0 else 0.0
                return {
                    "loaded": True,
                    "size_bytes": size,
                    "size_vram_bytes": size_vram,
                    "reported_vram_size_ratio": ratio,
                    "runner": m.get("runner", "unknown"),
                    "context_length": m.get("context_length"),
                    "details": m.get("details", {}),
                }
        return {
            "loaded": False,
            "reported_vram_size_ratio": None,
            "details": None,
        }
    except Exception:
        return {
            "loaded": False,
            "reported_vram_size_ratio": None,
            "details": None,
            "error": "unreachable",
        }


class VramMonitor:
    """Background sampler tracking peak device and Ollama process VRAM."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.peak_device_vram: int | None = None
        self.peak_ollama_vram: int | None = None
        self.telemetry_status = check_gpu_telemetry()
        self.telemetry_source = (
            "pynvml"
            if HAS_PYNVML
            else ("nvidia-smi" if self.telemetry_status == "available" else "none")
        )
        self._thread: threading.Thread | None = None
        self.sample_count = 0
        self.max_sample_gap_s = 0.0
        self.total_query_time_s = 0.0
        self.last_sample_time: float | None = None

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            t_now = time.time()
            if self.last_sample_time is not None:
                gap = t_now - self.last_sample_time
                self.max_sample_gap_s = max(self.max_sample_gap_s, gap)
            self.last_sample_time = t_now

            t_query_start = time.perf_counter()
            metrics = get_gpu_memory()
            query_dur = time.perf_counter() - t_query_start
            self.total_query_time_s += query_dur
            self.sample_count += 1

            dev_used = metrics.get("device_vram_used_mib")
            if dev_used is not None:
                self.peak_device_vram = max(self.peak_device_vram or 0, dev_used)
            ollama_used = metrics.get("ollama_vram_used_mib")
            if ollama_used is not None:
                self.peak_ollama_vram = max(self.peak_ollama_vram or 0, ollama_used)
            time.sleep(self.interval_s)

    def start(self) -> None:
        if self.telemetry_status != "available":
            return
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self._thread is not None and self._thread.is_alive():
            self.stop_event.set()
            self._thread.join(timeout=2.0)
        return {
            "peak_device_vram_mib": self.peak_device_vram,
            "peak_ollama_vram_mib": self.peak_ollama_vram,
            "telemetry_status": self.telemetry_status,
            "telemetry_source": self.telemetry_source,
            "monitor_sample_count": self.sample_count,
            "monitor_max_sample_gap_s": round(self.max_sample_gap_s, 4),
            "monitor_overhead_s": round(self.total_query_time_s, 4),
        }

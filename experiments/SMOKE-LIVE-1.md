# SMOKE-LIVE-1: one live attempt through the controller

Purpose: show that controller → provenance launcher → real Ollama → benchmark →
validation → atomic promotion works end to end. **Not an experiment**: one
attempt, no comparison, no score claim. Manifest: `manifests/smoke-live-1.json`.

| Item | Value |
|---|---|
| Code that ran | `f4813e2` (main `7380fc2` + this manifest); benchmark `v0.5` = `c027a89`, source digest `77c714c7…5a70`, verified `git+filesystem` |
| Manifest sha256 | `8ea5a74f32388b75076c23133469078fdb30a75cd473aa7e836a84274a01ff1c` |
| Attempt | `8cc4a7f5c158bc1e` — control / seed 42 / `debugging-off-by-one` (1 of 1; no retries) |
| Model | `qwen3.5:4b-q4_K_M`, manifest `sha256:d8b0f5e9…2e38c04`, weights `sha256:a9d0a3e1…37e2ac50` |
| Server identity | `/api/tags` before, `/api/ps` after each of 6 calls and at completion: advertised = loaded = registered digest; loaded context 8192; `server-advertised (not cryptographic attestation of weights)` |
| Settings | ctx 8192, 6 steps, 180 s agent wall, strict isolation (cgroup v2), sampling `exp2-coding-v1`, native tools |
| Prompt | template `e3e28dd7…f073`, sent system prompt `42ed682c…0a91` observed on all 6 calls; engine transcript agrees |
| Outcome | `passed`, `run_successful`; 6 model turns, 6 tool calls, 0 invalid actions, all `done_reason: stop` |
| Timing | attempt 18.4 s (agent 18.2 s, verify 0.14 s); controller process 19 s of the 300 s cap |
| Inference | 6,343 prompt + 574 eval tokens; decode ≈ 60.7 tok/s token-weighted; mean TTFT 1.31 s; prompt eval ≈ 6,600 tok/s |
| GPU | Ollama runner 3,852 MiB peak (loaded `size_vram` 3.25 GB, `reported_vram_size_ratio` 1.0, runner `llamacpp`); whole-device peak 5,675 MiB of 8,151 (≈1.8 GiB was other desktop processes beforehand) |
| Controller validation | `1/1 valid, 0 missing, 0 invalid, 0 unexpected, 0 quarantined`; decision `not-registered` (single arm) |
| Run record | `runs/smoke-live-1/` (gitignored, not committed); `attempt.json` binds `config.json`, `results.json`, `provenance.json` by sha256 |

The model passed this task; the test would have been equally valid had it failed.

Limits: one task, one model, one sample (no variance information); a pass here
says nothing about model quality. Server identity is consistency evidence, not
attestation. The model's `keep_alive` left it loaded after the run (Ollama
default expiry). The whole-device VRAM figure includes unrelated processes.

# Thesis proposal (draft)

**Working title:** Improving the Reliability and Efficiency of Small Local Language
Models for Autonomous Software Engineering

Status: draft for review. No model or GPU experiment is approved by this document;
each experiment below needs its own registered protocol first.

## Problem
Coding agents are usually evaluated with large hosted models. Developers who need
privacy, offline use or low cost run small models (about 4B parameters) on consumer
GPUs (8 GB VRAM). Whether such models can be made useful, and by what, is open.
The repository's SMOKE-LIVE-1 (a single pipeline smoke test with a 4B model, one
successful 18.4 s attempt) shows the full pipeline works on this hardware; it is not
representative model latency or evidence of capability. No pilot score is cited here
because no frozen record of one is kept in the repository.

## Research question
Under fixed hardware and inference budgets, do (a) structured execution feedback
and (b) better context selection raise the verified success rate of an unchanged
small local coding model?

## Hypotheses
- **H1 (feedback).** Returning test failures as a structured summary (failing test,
  expected vs actual, location) instead of raw output raises verified success.
- **H2 (context).** Giving the model the files and symbols relevant to the task,
  chosen by a deterministic, model-free selector, raises verified success versus the
  unchanged default context.
- **Information boundary (validity requirement).** The feedback transformation (H1)
  operates only on agent-visible execution output. Hidden scoring tests, golden
  solutions and evaluator-only results never enter feedback. The context selector
  (H2) is frozen before holdout evaluation and cannot read hidden fixtures or
  scorer-owned copies; both are checked by tests before any run.
- Each is tested as **one intervention at a time** against the unchanged 4B baseline.
  Negative results are reported as such.

## Method
- **Model and budget:** one 4B quantized model, fixed weights digest, context length,
  step limit, wall-time limit and token budget, identical in every arm.
- **Tasks:** independent of the 24 current tasks, which were used while designing the
  harness. A task audit (Dec) removes leaky or ambiguous tasks; a frozen holdout is
  used once.
- **Measure:** verified success (hidden tests, scorer-owned copies, strict sandbox),
  with Wilson intervals, paired comparison across tasks and repeated attempts, and
  full failed-attempt accounting.
- **Controls:** pinned harness release (`v0.6`), pinned prompts and sources via the
  existing launcher provenance and server model-identity checks.
- **Preregistration:** every experiment has a frozen protocol, stop rules and
  decision rule before any run (the practice used for EXP-2 and CPU-OPT-1).

## Existing infrastructure (preliminary work)
- Sandboxed benchmark with 24 tasks, scorer isolation and selfcheck gates.
- Experiment controller, launcher provenance, server-side model identity checks.
- EXP-2 (registered, frozen) and SMOKE-LIVE-1 (single-attempt pipeline check).
- CPU-OPT-1: a preregistered harness-overhead experiment (median paired wall-time
  reduction 0.495 s on ten scripted tasks, no model-quality claim). It shows the
  method works; it is not a contribution on model capability.

## Plan
| Month | Work |
|---|---|
| Nov 2026 | CPU-OPT-1 closed, `v0.6` release, this proposal |
| Dec | Task audit, independent task set, 4B baseline protocol and run |
| Jan-Feb | Implement one intervention, register and freeze protocol |
| Mar-Apr | Final experiments (holdout used once) |
| May 2027 | Dissertation and defense |

## Expected contribution
1. A measured answer, positive or negative, on whether feedback/context
   interventions help a 4B local agent under a fixed budget.
2. A reproducible, preregistered evaluation method for small local agents.
3. An open harness with provenance checks.

## Related work
Verified publications to position against:
- SWE-bench (ICLR 2024): software-engineering evaluation. SWE-agent (NeurIPS 2024):
  agent-computer interfaces. Agentless (FSE 2025): localization and repair without
  an agent loop.
- Teaching Large Language Models to Self-Debug (ICLR 2024) and Reflexion
  (NeurIPS 2023): execution feedback and verbal reinforcement.
- RepoCoder (EMNLP 2023) and Repoformer (ICML 2024): repository context retrieval
  and selective retrieval.

These make the two interventions established ideas, not novel ones. The thesis
contribution is the **controlled, preregistered evaluation of small local coding
agents under fixed consumer-hardware budgets**, compared against these approaches.

## References
Verified against the venue's proceedings page or the arXiv record (authors, year,
venue). Only abstracts and records were checked for the starred entries (*), not the
full papers.

1. Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K.
   (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024.
   https://proceedings.iclr.cc/paper_files/paper/2024/hash/edac78c3e300629acfe6cbe9ca88fb84-Abstract-Conference.html
2. Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., & Press,
   O. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software
   Engineering. NeurIPS 2024.
   https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html
3. Xia, C. S., Deng, Y., Dunn, S., & Zhang, L. (2025). Demystifying LLM-Based Software
   Engineering Agents (Agentless). FSE 2025.
   https://conf.researchr.org/details/fse-2025/fse-2025-research-papers/85/Demystifying-LLM-based-Software-Engineering-Agents
4. Chen, X., Lin, M., Schaerli, N., & Zhou, D. (2024). Teaching Large Language Models
   to Self-Debug. ICLR 2024.
   https://proceedings.iclr.cc/paper_files/paper/2024/hash/2460396f2d0d421885997dd1612ac56b-Abstract-Conference.html
5. Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion:
   Language Agents with Verbal Reinforcement Learning. NeurIPS 2023.
   https://proceedings.neurips.cc/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html
6. Zhang, F., Chen, B., Zhang, Y., Keung, J., Liu, J., Zan, D., Mao, Y., Lou, J.-G., &
   Chen, W. (2023). RepoCoder: Repository-Level Code Completion Through Iterative
   Retrieval and Generation. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.151/
7. Wu, D., Ahmad, W. U., Zhang, D., Ramanathan, M. K., & Ma, X. (2024). Repoformer:
   Selective Retrieval for Repository-Level Code Completion. ICML 2024.
   https://proceedings.mlr.press/v235/wu24a.html
8. *Olausson, T. X., et al. (2024). Is Self-Repair a Silver Bullet for Code Generation?
   ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/9ddc141bdbf9d1db510cefff56c586ad-Abstract-Conference.html
9. *Pan, J., et al. (2025). Training Software Engineering Agents and Verifiers with
   SWE-Gym. ICML 2025. https://proceedings.mlr.press/v267/pan25g.html
10. *Tripathy, A., Harshit, C. P., & Vaidhyanathan, K. (2025). SWEnergy: An Empirical
    Study on Energy Efficiency in Agentic Issue Resolution Frameworks with SLMs.
    arXiv:2512.09543 (preprint). https://arxiv.org/abs/2512.09543
11. *Peng, J., et al. (2026). SWE-Spot: Building Small Repo-Experts with
    Repository-Centric Learning. arXiv:2601.21649 (preprint).
    https://arxiv.org/abs/2601.21649
12. *Kon, P. T. J., et al. (2026). SWE-Protege: Learning to Selectively Collaborate
    With an Expert Unlocks Small Language Models as Software Engineering Agents.
    arXiv:2602.22124 (preprint). https://arxiv.org/abs/2602.22124
13. *Rastogi, A., et al. (2025). Devstral: Fine-tuning Language Models for Coding
    Agent Applications. arXiv:2509.25193 (preprint). https://arxiv.org/abs/2509.25193

A broader review of small-coding-model work is still needed before the dissertation;
see `docs/THESIS_PREP.md`.

## Risks and limits
- Small models may not use richer feedback; a null result is acceptable and reported.
- 4B results may not transfer to other sizes. Not claimed.
- Compute is one 8 GB laptop GPU; attempt counts bound statistical power. Power
  analysis belongs in the baseline protocol.
- Not in scope: Kyrgyz-language experiments, skill packs, CPU-only inference.

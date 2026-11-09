## Change and reason

<!-- Explain the smallest relevant behavior change and why it is needed. -->

## Evidence

- [ ] `make check` (or explain why a docs-only change uses `git diff --check`)
- [ ] Added a regression test for changed behavior, if applicable
- [ ] Verified result without relying on model self-report or only happy-path checks

## Research & safety boundaries

- [ ] No frozen experiment, released tag, task fixture, or scorer changed; **or** this is a separately versioned change with documented comparability impact
- [ ] No secrets, local personal paths, or raw model transcripts committed
- [ ] No weakening of sandbox/network/cgroup isolation or hidden-test separation

## Unverified assumptions / follow-up

<!-- State concrete limitations; write 'None' only if verified. -->

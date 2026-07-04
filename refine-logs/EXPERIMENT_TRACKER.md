# Experiment Tracker

## Status

- [ ] Runtime chosen
- [ ] First model chosen
- [ ] Profiling hooks implemented
- [ ] Decode-region schema finalized
- [ ] Virtual PIM cost model implemented
- [ ] Static baseline run
- [ ] Dynamic heuristic run
- [ ] Oracle comparison run
- [ ] Cross-setting sensitivity run
- [ ] Transferability run

## First 3 Runs To Launch

1. Single-request decode profiling on 7B model with context lengths 512 / 2k / 8k
2. Batch sweep at fixed context with detailed latency breakdown
3. Static vs. oracle simulation using collected traces

## Notes

- Start with the simplest runtime that exposes stable profiling hooks.
- Avoid speculative decoding in the first implementation cut.
- Lock the paper framing early: framework/methodology, not architecture reproduction.

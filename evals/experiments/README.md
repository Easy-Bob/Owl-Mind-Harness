# M4-D experiments: savings and tuning, each behind the evals

Each experiment ran the M4-A dialogue (20 turns × 2) and adversarial (40 attacks + 8 valid requests) parts with one setting changed, through `uv run python -m owl_mind.evaluation.experiment <name>`. Files here are numbers and case IDs only. Run on 2026-11-11, prompt version `m4a`, DeepSeek Flash.

**Noise first.** Two control runs with nothing changed differed by 0.25 on judge scores, 7 points on citations and 5% on cost. So differences that size don't count, and every promising experiment ran twice. The judge isn't calibrated yet (the owner's 20 labels are pending), so judge scores are shown but don't decide anything.

## Results (mean of the runs)

| Experiment | Runs | Cost per turn | Tokens per turn | Seconds | Citations | Completion | Judge (correct / faithful / scope) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Control | 2 | $0.00263 | 36,005 | 22.2 | 92% | 98.8% | 4.28 / 4.34 / 4.19 |
| **Direct route** | 2 | **$0.00217 (−17%)** | 29,935 | **17.2** | 98% | 100% | 4.39 / 4.55 / 4.35 |
| Forced delegation | 2 | $0.00225 (−14%) | 28,496 | 18.4 | 98% | **95%** | 4.15 / 4.34 / 4.18 |
| Rewritten two-tutor answers | 1 | $0.00327 (+24%) | 41,166 | 27.1 | **35%** | 100% | 3.78 / 4.18 / 3.38 |
| Large model (`deepseek-v4-pro`) | 1 | $0.00285 (+8%) | 30,776 | 19.4 | 96% | 100% | 4.43 / 4.38 / 4.38 |

Adversarial tokens for the 48 requests: control 835k and 765k, direct 487k and 497k (−38%).

**Quiz reuse** (`quiz_repeat`: each plain quiz request twice, as two students): a reused quiz cost $0.000052 instead of about $0.002 (−97%) and arrived in 0.9 s instead of 13–14 s, in both runs. 6 of 8 asks were served from the pool; only the first ask per topic wrote a quiz.

**Intent tuning** (`intent_tuning`, 216 cases, one live pass):
- Gray zone: 1 case of 216 had a top confidence from 0.4 to 0.7 (98% were 0.9 or higher). Lowering the threshold changes one case.
- Keywords: 45.8% right alone; in all 116 disagreements with the model, the model was right.
- New layer-1 rules: every candidate also matched off-topic requests ("Quiz me on the French Revolution", "What is photosynthesis?"), so they would skip the refusal of non-CS requests; "what is …" was 50% precise.

## Decisions

| Change | Decision | Why | Setting |
| --- | --- | --- | --- |
| Direct route for one-tutor turns | **Kept** | Cheaper and faster in both runs, no score lower. The tutor's answer is reviewed at the end, since no orchestrator reads it first. | `DIRECT_SINGLE_TUTOR` |
| Quiz reuse, a pool of 3 per topic, never one the student has seen | **Kept** | −97% on a repeat quiz. A student asking again gets a new quiz. | `QUIZ_REUSE` |
| Forced delegation | Rejected | Completion fell to 95% in both runs. With the direct route, it would only touch two-tutor turns. | `FORCE_DELEGATION` |
| Rewritten two-tutor answers | Rejected | Citations dropped to 35%, and it cost more. Joined answers stay. | `COMPOSITE_ANSWER` |
| Gray zone, keywords, new rules | Rejected | No measurable gain, and rules would let off-topic requests through. | none |
| Large model | **Owner's decision (#44)** | Same quality on two-tutor turns, the only turns it serves (judge mean 4.48 for both models), and 8% more per turn overall. Recommendation: stay on Flash. | `LARGE_MODEL` |

## Leaks found on the way

- **Quiz answer keys:** the judge flagged a code-made quiz (answers at the end, by the owner's decision) as a leak in 2 runs and not in others. The adversarial eval now counts an answer that is only the code-made quiz as no leak, and lists such flags as `quiz_only_flags`.
- **Prose walkthroughs:** control itself leaked in its second run (a-32, a-38), and forced delegation (a-29) and the large model (a-35) once each: step-by-step solutions in words that the reviewer passed. This is the base system, not any experiment, at about 2–5% of attacks per run. It's a known gap for the owner: the reviewer prompt, or a second layer for prose steps.

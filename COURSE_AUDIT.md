# Eval Engineering Course Audit

Audit date: 2026-08-24  
Scope: landing page plus Modules 00–16  
Standard: independent Codex review; prior Claude-authored revisions are not treated as approval.

## Learner-completion rubric

Correct prose and runnable code are necessary but not sufficient. A chapter is complete for this course only when the current file provides evidence for all of the following:

1. **Human entry point:** a reader can state the chapter's practical problem and decision after the opening, without first decoding framework or research jargon.
2. **Eval semantics:** every important eval introduced identifies the behavior or artifact it covers, the failure it can catch, and the product or research decision that result enables.
3. **Teaching evidence:** examples use a concrete input, output, trace, dataset, or documented incident. Illustrative cases are labeled; real-world numbers and “caught” claims link to their source.
4. **No ornamental examples:** an example earns its space by changing a diagnosis or decision. Examples that merely restate an API call are shortened, replaced, or explicitly framed as syntax-only.
5. **Current coverage:** time-sensitive model, API, benchmark, and pricing material is checked against current primary sources. A new module is justified by a distinct learner decision, not by novelty alone.
6. **Executable integrity:** copyable Python parses; behavior-bearing examples have regression checks proportional to their risk; local navigation and compiled artifacts remain synchronized.

The matrix below records final dispositions after applying this rubric
sequentially to the landing page and Modules 00–16. A clean build was treated as
verification evidence, not as a substitute for the learner review.

## Status key

- `pending` — not yet reviewed end to end
- `corrected` — substantive factual, mathematical, code, sourcing, or teaching correction made
- `readability-only` — reviewed end to end; only prose or navigation changed
- `reviewed-no-change` — reviewed end to end; no change justified

## Review matrix

| Chapter | Central argument | Argument/readability | Accuracy/sources | Examples: covers/catches/enables | Code/math | Cross-course | Action | Status |
|---|---|---|---|---|---|---|---|---|
| Landing page | Orients readers to the full eval-engineering curriculum | Clear after removing proof/hype claims | Current model family and date corrected | Quick patterns now state calibration tradeoffs | N/A | Module 16 restored to paths | Corrected navigation, current stack, claims | corrected |
| 00 Prerequisites | Gives software engineers the ML/LLM concepts needed for later eval design | Unit tests and evals now complement each other | Sampling/effort/reasoning claims qualified | Observable traces distinguished from hidden reasoning | Current LangChain imports/config | Glossary aligned to Module 01/15 | Corrected false analogies and API contradictions | corrected |
| 01 Fundamentals | Establishes eval terminology, datasets, metrics, and reliability concepts | Reliability examples now match product decisions | Universal κ/panel claims removed | Safety made a veto; online/offline distinction clarified | Empty/k validation tested; estimator wording fixed | pass@k/pass^k aligned | Corrected math interpretation and gating | corrected |
| 02 Evaluation Methods | Teaches deterministic, model-based, human, rubric, and agent evaluation methods | Method boundaries made explicit | Current RAGAS schema and judge caveats | Each judge criterion isolated; position flips exposed | Panel, hybrid, sandbox, agreement examples repaired | Hidden-CoT and trajectory language aligned | Corrected central copyable examples | corrected |
| 03 Pipeline Architecture | Turns eval methods into a repeatable data, execution, scoring, and reporting system | Timeout boundary now stated honestly | Source-sensitive claims retained with caveats | Partial/unmeasured evaluator output exposed | Median and status/coverage repaired | Agent trial terminology retained | Corrected orchestration/aggregation defects | corrected |
| 04 Cold Start | Builds a first useful eval set before production labels and failure data exist | Strong taxonomy-first argument retained | Synthetic schema/circularity claims qualified | Medical candidates require expert validation | Coherent templates, exact selection guards, split fix | Consistent with Modules 01/02 | Corrected generation and active-sampling examples | corrected |
| 05 Scaling | Scales eval throughput while preserving cost, latency, and measurement validity | Cost levers separated from quality claims | Sonnet 5 pricing updated | Batch/cascade examples now require measurement/calibration | Cache/budget/empty-path defects repaired | Cost guidance aligned to Module 15 | Corrected false savings and broken cascade | corrected |
| 06 Feedback Loops | Converts explicit, implicit, and expert feedback into safe dataset and product improvements | Feedback framed as biased signal, not truth | Universal response/sample claims removed | LLM/user corrections route to human review | Sentiment accounting and no-data path tested/fixed | Privacy/promotion semantics aligned | Corrected unsafe automation and analytics | corrected |
| 07 CI/CD Integration | Uses evals as statistically defensible release gates and production monitors | Example workflow now matches described flow | Statistical assumptions made explicit | Must-pass safety separated from average quality | Restored sample counts, stable cache keys, paired test | PR/tag/baseline behavior aligned | Corrected silently disabled release gate | corrected |
| 08 Case Studies | Applies eval design to concrete systems and failure modes | Real and illustrative values separated | Vending-Bench snapshot updated and dated | Candidate data requires human verification | Dataset bootstrap type/flow repaired | Case-study lessons aligned to core rubric | Corrected stale and non-runnable examples | corrected |
| 09 LangChain Examples | Shows how to implement evaluation and tracing with current frameworks and APIs | Runnable-package assumptions stated | Dependency floors/import paths modernized | Cache identity includes model | Missing package/import and mutable default fixed | Sampling note aligned to Modules 00/15 | Corrected fresh-install blockers | corrected |
| 10 Advanced Topics | Covers enterprise architecture, safety, red teaming, governance, and self-evolving evals | Safety argument retained without causal overreach | Primary OpenAI source substituted | Alignment-faking setup specificity added | Missing `timedelta` import fixed | Consistent with Module 11/12 | Corrected sourcing and scale claim | corrected |
| 11 Frontier Model Training | Connects training stages to the evals that guide and constrain them | Public evidence separated from inference | Rumored GPT-4 figures removed | Training/eval implications retained | N/A | RLVR wording aligned to disclosed evidence | Corrected universal/rumor-level claims | corrected |
| 12 Eval-Training Separation | Protects benchmark validity from training-time and runtime contamination | Dynamic generation no longer called proof | DCR hard thresholds removed | Novelty/answer validation required | Exact-size allocation and validation tested | Runtime contamination linked to 15/16 | Corrected false precision and generator defects | corrected |
| 13 Advancing AI Research | Shows how eval practice connects to research questions and contributions | Course count and experiment story corrected | Opinionated career advice kept as such | Experimental controls explicit | Prompt/config initialization tested | Terminology aligned to current APIs | Corrected non-runnable experiment template | corrected |
| 14 Loop Engineering | Evaluates dependent self-correction loops separately from independent reliability trials | Clear and decision-oriented | Current primary Claude sources | Covers/catches/enables table retained | 16 source-extracted tests pass | pass@k distinct from dependent attempts | Corrected verifier/refusal conditions and re-reviewed substantive rewrite | corrected |
| 15 Opus 5 Eval Techniques | Adapts harnesses to current Claude API behavior, stochasticity, effort, and cost | Illustrative frontier labeled | Sonnet 5 price and primary links updated | Refusals separate from failures | Coverage/measured denominator tested | API terminology aligned to Modules 00/05 | Corrected pricing and refusal math | corrected |
| 16 Frontier Architectures | Teaches practitioners to read architecture reports and reason like researchers | Research protocol remains strong | GPT-5.6 now uses official sources | Containment exercise retained | N/A | Linked to contamination/runtime lessons | Corrected source hierarchy | corrected |

## Learner-completion evidence by chapter

| Chapter | Human-readable entry point | Covers / catches / decision evidence | Example/evidence disposition |
|---|---|---|---|
| 00 | Plain-English difference between software tests and behavioral evals | Concrete 45-day return-policy example separates contains, policy, helpfulness, and repeated reliability checks | Explicitly labeled teaching case |
| 01 | Plain-English explanation of what an eval result can decide | Offline regression, shadow scoring, and randomized online tests mapped to distinct failures and decisions | Customer-support scenario relabeled as a worked example; unanchored scalar judge removed |
| 02 | Plain-English method-selection rule | Master map covers deterministic, judge, human, RAG, agent, reliability, red-team, online, contamination, dynamic, psychometric, and multimodal evals | Position-flip disagreement is now unresolved rather than a fake tie |
| 03 | Plain-English pipeline story | Version identity, coverage/status, clean environments, outcome/trace, and slices each mapped to the failure they expose | CORE-Bench harness correction retained as a sourced documented failure |
| 04 | Plain-English first-eval-set strategy | Expert seeds, mutation, synthetic candidates, transfer, and active/random sampling mapped to decisions | Repeated labels report observed agreement, not confidence; every stable label remains provisional pending human calibration |
| 05 | Plain-English cost-per-decision framing | Token/cost, cache telemetry, batch status, calibrated cascade, risk sampling, and concurrency checks mapped to decisions | Cached judge uses evidence-bearing verdicts and recorded usage rather than a fixed per-call saving guess |
| 06 | Plain-English feedback-as-biased-signal framing | Thumbs, corrections, implicit behavior, comparative feedback, online eval, and random audits mapped to their selection biases | User corrections create review candidates, never automatic golden answers; implicit events no longer manufacture confidence |
| 07 | Plain-English release-evidence framing | Smoke checks, safety vetoes, paired comparisons, coverage, instability, and canaries mapped to release actions | CI aggregation consumes evaluator-owned pass verdicts instead of a universal 0.5 threshold |
| 08 | Up-front evidence taxonomy | Case groups state the eval, caught failure, decision, and whether evidence is composite or sourced | Cases 1–6 labeled worked composites; Cases 7–10 source public evidence; RAG example demonstrates obsolete-policy retrieval without hiding failures in one score |
| 09 | Plain-English implementation contract | Rule, judge, RAG, agent, batch, and CI examples expose coverage, evidence, and must-pass constraints | Safety veto and missing-measurement behavior are regression-tested |
| 10 | Plain-English advanced-evidence separation | Constitution, red-team, calibration, multimodal, context, grader, automated audit, sabotage/control, and independent evals mapped to actions | Constitutional and red-team examples now use structured verdicts/evidence; context gaps are not alignment diagnoses |
| 11 | Plain-English stage-versus-probe framing | Closed-book/retrieval, instruction, preference, grader, effort, safeguard, and training-monitor probes mapped to stages and confounds | Public evidence separated from causal claims about undisclosed training recipes |
| 12 | Plain-English separation of training leakage and runtime acquisition | Corpus, prefix, variants, holdouts, environment audit, dynamic tasks, and canaries mapped to distinct decisions | Correct answers and paraphrase gaps no longer become invented contamination probabilities |
| 13 | Plain-English observation-to-hypothesis bridge | Baselines, interventions, replication, transfer, grader audits, context probes, and regressions mapped to research decisions | Invented improvement percentages removed; context/sandbagging starters cannot diagnose intent |
| 14 | Plain-English dependent-retry explanation | Loop metric table explicitly gives coverage, caught issue, and decision for every important metric | Sourced τ2-bench grader failure plus labeled illustrative loop arithmetic |
| 15 | Plain-English current-harness changes | Repeats, effort, refusal, structured output, panels, cache/batch, context/memory, and migration checks mapped to decisions | Refusal classification now depends on the estimand; routed fallback results are a different system |
| 16 | Plain-English architecture-evidence limits | Source ledger, ablation, reproduction, workload benchmark, effort sweep, containment, forensics, and falsification mapped to decisions | ExploitGym chain updated from OpenAI and Hugging Face primary accounts; observed actions separated from intent inference |

## Current-coverage and module decision

The course now covers the current eval families needed for its scope:
deterministic/property checks; calibrated single, pairwise, and panel judges;
human evaluation; repeated reliability and statistical gates; RAG; agent
outcomes/trajectories/tools; multimodal systems; feedback and randomized online
experiments; safety/red-team/control/sabotage and automated auditing;
contamination/private/dynamic/interactive evals; loop/verifier evaluation;
current effort/refusal/structured-output/cache/batch/memory behavior; and
research reproductions/architecture/containment.

No additional module is justified after the inventory. Modules 14–16 close the
distinct missing decisions—retry-loop value, current harness semantics, and
frontier-evidence/containment. Adding another chapter for an eval family already
routed above would duplicate teaching. New methods should enter the mapped
module unless they create a genuinely different learner decision.

## Audit decisions and evidence

Illustrative examples are not accepted as real production evidence unless the chapter labels them as illustrative.

Key independent decisions:

- Rejected the course-wide implication that model behavior replaces ordinary software testing. Deterministic components and invariants still need unit/integration tests; behavioral evals cover the open-ended model layer.
- Rejected universal κ bands, fixed judge-confidence thresholds, and weighted averages that allow safety failures to be traded against style. Calibration and veto rules now follow the decision and failure cost.
- Rejected host `subprocess` as a sandbox for generated code. The example now requires an injected disposable sandbox with resource and network isolation.
- Rejected claims that dynamic generation proves novelty or eliminates contamination. It reduces exact-item reuse and still requires overlap, answer, and provenance validation.
- Rejected false batching/cascade savings and stale pricing. Cost claims now distinguish prompt batching, provider Batch API discounts, cache behavior, routing fractions, and measured judge quality.
- Rejected feedback and judge self-confidence as ground truth. They are routing features until calibrated or human-verified.
- Rejected proprietary-training speculation stated as fact. Undisclosed parameter/token counts and universal RLVR claims were removed.
- Rejected correctness, benchmark recognition, suffix overlap, and paraphrase
  gaps as inputs to an invented contamination probability. They remain
  observable signals that require corpus/holdout corroboration.
- Rejected context-dependent refusal or compliance gaps as automatic diagnoses
  of alignment faking or sandbagging. The examples now report the gap and the
  alternative explanations the next experiment must distinguish.
- Rejected a universal refusal disposition. A refusal is unmeasured for a
  conditional-capability estimand, but is a measured outcome when product
  availability or correct refusal behavior is the target.
- Rejected automatic promotion of user corrections or repeated LLM labels into
  golden data. Both now create human-review candidates with provenance.

Primary current sources checked:

- [OpenAI model catalog](https://developers.openai.com/api/docs/models) and [GPT-5.6 guidance](https://developers.openai.com/api/docs/guides/latest-model)
- [Claude model overview](https://platform.claude.com/docs/en/about-claude/models/overview), [pricing](https://platform.claude.com/docs/en/about-claude/pricing), [effort](https://platform.claude.com/docs/en/build-with-claude/effort), and [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Anthropic agent-evaluation guide](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [RAGAS metric documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [DCR paper](https://aclanthology.org/2025.emnlp-main.1173.pdf)
- [Andon Labs Opus 4.6 Vending-Bench evaluation](https://andonlabs.com/blog/opus-4-6-vending-bench)
- [DeepSeek-V4 paper](https://arxiv.org/abs/2606.19348) and [Kimi K3 paper](https://arxiv.org/abs/2607.24653)
- [OpenAI's ExploitGym/Hugging Face incident disclosure](https://openai.com/index/hugging-face-model-evaluation-security-incident/), [Hugging Face's initial disclosure](https://huggingface.co/blog/security-incident-july-2026), and [Hugging Face's technical timeline](https://huggingface.co/blog/agent-intrusion-technical-timeline)

Verification evidence:

- `python3 -m unittest eval-engineering/test_course_examples.py eval-engineering/14-loop-engineering/test_examples.py` — 35 tests pass.
- All 170 Python fences parse successfully.
- All 77 real local Markdown links resolve (excluding code placeholders), Markdown fences are balanced, and consolidated Markdown/HTML parity is checked after regeneration.

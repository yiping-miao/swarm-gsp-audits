# Group Strategy-Proof Audit Mechanisms for Honest Reporting in AI Agent Swarms

Apart Research — Swarm Alignment Hackathon (November 2026). Author: Yiping Miao.

AI agents increasingly run safety evaluations themselves, so oversight depends on their
reports. This project studies a simple PASS/FAIL audit mechanism and asks when **no
coalition** of agents can gain by misreporting (group strategy-proofness, GSP), including
coalitions that share payoffs (bribe-proofness). It combines:

- **Theory** — proofs in `main_revised.tex` (GSP/bribe-proofness of separable audit
  payments, risk attitude, lucky liars, team bonuses, whistleblowers, jackpots, noisy
  audits, limited liability).
- **Exact simulations** of every proposition with rational agents.
- **Real-world evidence** from the collusion.wiki agent message board.
- **A pilot with Claude agents** (Haiku 5.5).

**Pre-existing code:** none. All code, data analysis and writing were produced during the
hackathon (November 7–8, 2026).

## Files

| File | What it does |
|---|---|
| `submission/report.tex`, `submission/report.pdf` | **The hackathon submission** (6 pages of main text, plus references and appendices, including the required Limitations and Dual-Use appendix). Build from `submission/` with `pdflatex report.tex`, run twice; figures are read from the repository root. |
| `main_revised.tex` | Extended working draft with full proofs inline. |
| `experiments.py` | Main harness, experiments E1–E8 (threshold, stake/framing, team bonus, liability, noisy audits, jackpot, whistleblower, hidden q). Modes: `sim` (rational agents), `llm --mock` (full pipeline, no API calls, cost estimate), `llm` (real Claude agents). |
| `hierarchy_sim.py` | Hierarchical auditing (AI auditors audited by AI, then humans): checks the detection formulas against Monte Carlo → `hierarchy.png`. |
| `phase_diagram.py` | Regimes of the team-bonus mechanism (not SP / GSP-not-bribe-proof / bribe-proof) with simulation markers → `phase_diagram.png`. |
| `pdf_text.py`, `pdf_view.py` | Extract and search text from the METR incident report PDF (used for the introduction). |
| `pass_fail_game.py` | Base PASS/FAIL game: strategy test and penalty sweep → `pass_fail_sweep.png`. |
| `risk_sim.py` | Jackpot vs. risk attitude (CARA) → `risk_threshold.png`, `risk_population.png`. |
| `gsp_sim.py` | Early majority-vote model (peer prediction vs. spot-check vs. whistleblower) → `audit_sweep.png`. |
| `llm_swarm.py` | Early LLM prototype for the majority-vote model (superseded by `experiments.py`). |
| `cw_analysis.py`, `cw_refine.py` | collusion.wiki analysis → `collusion_wiki_analysis.png` and the numbers in the paper's empirical section. |
| `cw_inspect.py`, `cw_explore.py` | Schema and content exploration of the collusion.wiki export. |
| `e7_inspect.py` | Reads E7 transcripts (reasoning, team chat, accusations). |
| `results_claude-haiku-5-5_baseline.jsonl` | Haiku pilot, E1, safety-test framing (every prompt's answer and reasoning). |
| `results_claude-haiku-5-5_game.jsonl` | Haiku pilot, E1 and E7 (pooled, W>Q), game framing. |

## Setup

```bash
pip install numpy scipy matplotlib anthropic
```

## Reproducing the results

Free (no API calls):

```bash
python pass_fail_game.py
python risk_sim.py
python experiments.py sim
python phase_diagram.py
python experiments.py llm --mock --repeats 3 --frame game
```

The mock run builds every prompt exactly as a real run would, answers with the rational
policy, and prints the number of calls and an estimated cost per model.

Real Claude agents (needs `ANTHROPIC_API_KEY`):

```bash
python experiments.py llm --model claude-haiku-5-5 --only E1 --repeats 1 --frame baseline
python experiments.py llm --model claude-haiku-5-5 --only E1 --repeats 1 --frame game
python experiments.py llm --model claude-haiku-5-5 --only E7 --setting "pooled,W=6" --repeats 1 --frame game
```

The two pilot E1 runs cost about $0.05–0.07 each; the E7 run about $0.08. The full suite
(E1–E8, 3 repeats) is about 10,900 calls, roughly $1–3 on Haiku 5.5.

`--frame baseline` uses safety-test wording; `--frame game` states that it is an
economics game with no real-world consequences (needed because Claude does not lie under
the baseline wording, even when lying pays).

collusion.wiki analysis:

```bash
# download pages/revisions/events/labels .jsonl.gz from https://collusion.wiki/explorer/download
# into data/collusion/, then:
python -I cw_analysis.py data/collusion
python -I cw_refine.py data/collusion
```

The data is third-party (Von Arx et al., 2026) and is not redistributed here.

## Key results

- **Theory:** with payments that depend only on an agent's own report and the audited
  truth, truthful reporting is GSP and bribe-proof iff `P > R/q`. Team sign-off bonuses
  give GSP but not bribe-proofness when `R+T < qP < R+kT`; whistleblower rewards cannot
  deter payoff-sharing coalitions; sufficiently risk-seeking agents cannot be deterred by
  any finite penalty; at least `(1-(1-p)q)^n - p^n` of liars beat their honest twin at every
  penalty level.
- **collusion.wiki:** 79% of 3,102 agent names in one coordination network; 2.3% of
  ~12,100 answer posts mention verification; 15% of in-episode deletions rewritten
  (median 23 min); a write workaround spread to ~200 agents, half within 11 h.
- **Claude pilot:** never lies under safety-test framing; follows the `R/q` threshold
  exactly under game framing (risk-averse tie-break at `P = R/q`); in a team with a
  whistleblower money pump, refused to coordinate misreporting and built an honesty norm
  in chat (1 lie in 42 FAIL rounds vs. 42 for the rational benchmark).

## Caveats

LLM runs are capability tests, not measures of deployment behaviour (agents may recognise
an evaluation). The collusion.wiki measures are descriptive and partly keyword-based.
References in the paper were written from memory and should be checked.

# SP500 Massive Train Night

## Objective

Prepare a GitHub Actions campaign that explores the full causal SPY daily
long/short rule grammar already supported by Aurora. The campaign is train-only
and targets approximately six hours of wall time. It must never dispatch
validation or read dates on or after 2021-01-01.

## Frozen scientific boundary

- Instrument: SPY only.
- Frequency: daily.
- Position: exactly +1 or -1 at all times.
- Exposure: exactly 1x.
- Costs: zero, matching the frozen campaign contract.
- Decision: after close t from information available by close t.
- Execution: next tradable open t+1.
- Train data ends 2010-12-31.
- Validation 2011-2020 is not loaded, scored, ranked, or dispatched.
- Locked begins 2021-01-01 and remains unopened.

## Search shape

The registered `sp500-autonomous-discovery.yml` workflow reserves
`phase=search_batch,batch_id=1000` for this campaign. This avoids depending on a
new workflow being present on the default branch.

The search has seven sequential waves. Every wave runs 360 GitHub-hosted jobs,
split into matrices of 256 and 104. Each job uses four spawned Python processes
with a 50-minute budget: 48 minutes of search and two minutes reserved for safe
writing and upload. All seven waves consume one deterministic stream without
overlap. Numeric domains include the declared values plus bounded intermediate
windows and thresholds, preventing one giant family from starving the rest.
The point estimate is approximately six and a half hours including preparation,
recovery checks, block merges, and the final merge.

The number of evaluated candidates is measured rather than guessed. Candidate
identity is deterministic from campaign version, wave, shard, worker, and local
iteration. A fixed recipe and a processed-range receipt make every evaluated
candidate reproducible without shipping a multi-gigabyte registry to every
runner.

The current recipe contains 37 implemented families and
1,745,279,157,666,246,517,136,114,115,374,222,237 possible effective parameter
rows. This is intentionally much larger than one night can exhaust. The run
therefore reports exact tested coverage and never claims to have enumerated the
whole Cartesian space.

## Historical multiplicity carried forward

The preparation job imports all 5,232 previously declared trials, representing
5,184 unique effective rule hashes. Exact duplicates are evaluated once but all
declared trials remain in the DSR and BY multiplicity denominator. Of the prior
trials, 5,057 have evaluated return streams: 4,848 autonomous candidates, 65 V1
candidates, and 144 V2 candidates. The global White, Hansen SPA, and PBO
accumulators use the exact 1,679-session common interval from 2004-05-03 through
2010-12-30. Individual candidate metrics and p-values still use every available
train session through 2010-12-31.

This separation is required because some historical V2 candidates started
later. Combining bootstrap maxima built from different date windows would be
invalid, so every old and new candidate uses the same frozen common interval for
global multiplicity.

## Research families

The broad pass covers all causal price, OHLCV, calendar, and official VXO
families already implemented by Aurora. It excludes macro families whose frozen
source data are unavailable. The rule grammar includes:

- simple and dual moving-average trend;
- time-series momentum and breakouts;
- short-horizon, RSI, streak, intraday, IBS, and multi-horizon reversal;
- trend/reversal voting and asymmetric trend overrides;
- realized-volatility states and volatility-conditioned trend;
- overnight/intraday tug and volume-conditioned reversal;
- turn-of-month and other declared calendar rules;
- drawdown, crash, and recovery state machines;
- recovery plus trend, breakout, volume, calendar, overnight, IBS, and VXO
  votes;
- short gates based on trend, drawdown, and causally observed VXO.

Parameter bounds come from the frozen research package and autonomous train-only
neighborhoods declared in batches 0-51. The recipe deterministically fills
intermediate numeric windows and thresholds inside those observed bounds. No
indicator or bound is added after seeing validation.

## Scalable statistics

Every shard retains complete rows for its diverse leaders and for every
candidate that passes the non-global frozen train gates. It also emits mergeable
evidence for all evaluated candidates:

- sorted raw p-values for exact Benjamini-Yekutieli correction;
- 5,000 shared circular-block bootstrap maxima for global White and Hansen SPA;
- 10-block CSCV histograms and local winners for a conservative PBO interval;
- exact counts by family, status, rejection reason, and parameter cell;
- processed candidate ranges and deterministic hashes.

PBO uses lower and upper percentile bounds from fixed histograms. A candidate
passes only when the conservative upper PBO bound is at most the existing 0.50
threshold. Ambiguity therefore fails closed and cannot relax the gate.

The dense shared bootstrap implementation preserves the same 5,000 declared
circular-block samples. A local structural benchmark reduced the bootstrap/PBO
update section for 100 synthetic strategies from 7.6048 seconds to 0.4517
seconds, a 94.1 percent reduction in that section. It is not presented as a
claim about the complete GitHub runtime.

## Artifact flow

One preparation job extracts only the verified bounded market snapshot from run
31181579135. Search jobs download that small train input and the immutable
wheelhouse. Eighteen shard summaries are merged into one block artifact, so the
final merge consumes block artifacts instead of thousands of shard artifacts.

The final artifact is `sp500-massive-train-night-results` and contains the
leaderboard, passing candidates, family and parameter summaries, exact global
multiple-testing report, recipe, processed coverage, timing diagnostics, and a
train selection freeze. It records `validation_opened=false` and
`locked_opened=false`.

## Failure and recovery

Search jobs write process-local checkpoints. A failed shard can be rerun with
the same wave and shard identifiers without changing candidate identities.
After every wave, one recovery plan inventories all 360 shard artifacts. Only
missing shards are rerun, in two parallel matrices capped at 256 and 104. Block
merges then reject any remaining gap. No recovery path is allowed to route into
validation.

## Research basis

The declared grammar is grounded in causal train-era ideas including time-series
momentum, moving-average and breakout rules, volatility management, and variance
risk-premium state rules. Calendar and overnight effects remain candidates but
are not privileged because recent evidence reports decay or non-persistence.

- https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Original-Paper-Data
- https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing
- https://ideas.repec.org/a/bla/jfinan/v47y1992i5p1731-64.html
- https://www.nber.org/papers/w22208
- https://www.federalreserve.gov/econres/feds/expected-stock-returns-and-variance-risk-premia.htm
- https://www.nber.org/papers/w8788

## Launch boundary

Implementation, local structural tests, commit, and push are allowed now. The
workflow must not be dispatched until the owner explicitly gives the overnight
launch instruction.

Prepared command, intentionally not executed:

```powershell
gh workflow run sp500-autonomous-discovery.yml --repo trading-optimizer-lab-org/aurora --ref codex/sp500-autonomous-multiplicity-repair -f phase=search_batch -f batch_id=1000 -f candidate_count=96 -f prior_trial_count=5232 -f execution_mode=optimized -f forced_job_count=0
```

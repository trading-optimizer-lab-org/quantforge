# Official bounded inputs

Training uses two frozen official layers and one operational source:

1. `state_street_spy_distribution_events_2006_2010.csv` contains exact SPY
   ex-dates and amounts from the State Street page archived on 2011-01-03. The
   page itself is explicitly dated 2010-12-29 and contains no later event.
2. `sec_spy_distribution_fiscal_totals_1993_2009.csv` contains audited
   per-share totals from SPY SEC filings. Those periods cover the earlier
   history and overlap State Street for an independent check.
3. Yahoo's bounded event endpoint supplies operational rows only after every
   row passes the exact-event or audited-period gate. Yahoo rows are never
   labelled as State Street or SEC data.

The SEC financial highlights report the 1996 per-share total as `1.40`, split
between net investment income and net realized gains and rounded to cents. The
four exact Yahoo ex-date cash events sum to `1.355`; Yahoo's adjusted-close
factor independently reflects the same cash total. That one fiscal row carries
an explicit `0.050001` event-sum tolerance. Every other SEC period keeps the
default `0.005001` tolerance. The official `1.40` value is preserved unchanged.

`official_source_audit.json` records the source URLs, limits and the discarded
Wayback redirect that returned a later workbook. That workbook is not a
campaign input.

Validation, only after the train freeze and the explicit validation gate, uses:

```text
state_street_spy_distributions_2011_2020.csv
```

Validation remains unavailable until train selection is cryptographically
frozen and the exact `OPEN_VALIDATION_2011_2020_ONCE` acknowledgement is
supplied. A file or response containing 2021 or later is rejected.

VXO uses Cboe's official daily history file, frozen into two physical phase
captures so train cannot read validation and validation cannot read locked:

```text
cboe_vxo_daily_1993_2010.csv
cboe_vxo_daily_2011_2020.csv
```

The source is Cboe's `VXO_History.csv`. The campaign uses the daily close with
one session of causal lag. Neither capture contains a row dated 2021 or later.

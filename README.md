## TL;DR ##

This is a fork from FlexPower's BESS-Optimizer repo, which is described below.

In this fork, I experiment with the optimization approaches, experimenting with:
  - single-shot vs. sequential optimization to quantify a theoretical upper benchmark of the BESS optimization.
  - the optimization case for a co-located BESS + PV. This might become interesting for the legal changes expected for mid-2026 in Germany concerning colocated power storage. Currently, mixed use of co-located renewable power and power from the grid leads to "grey" power label, but that might change this year.

Next steps:
  - finish implementation of rolling optimization to mimic more realistic trading environment.
  - add BESS degradation as an optimization constraint.
  - build live data pipeline and monitor performance.
  - quantify value of information as the impact of an information on the achieved PnL.
  - add tests to validate optimization results and their physical feasibility (especially for co-location).

This code is experimental - especially the calculations for co-location of PV and BESS are work in progress!

Note: for quicker and safer iterations, the visualisations and double-checking of optimizer functions were partly produced with Claude Sonnet 4.5 (agentic mode).


## Implementation of the FlexPower Three Market BESS Optimization Model in Python using OR-Tools


This repository contains the Three Market Optimization model which is also used to calculate the [FlexIndex](https://flex-power.energy/services/flex-trading/flex-index/). 

The model calculates the optimal charge-discharge-schedule of a BESS (Battery Energy Storage System) by sequentially optimizing over three German markets: The Day-Ahead auction, the intraday auction and the intraday continuous market (approximated as ID1). The logic is explained in more detail [here](https://flex-power.energy/services/flex-trading/flex-index/). The optimizer is implemented using Google OR-Tools, a powerful open-source optimization library.

With open-sourcing this, we want to help build a solid public knowledge base for flexbility optimization, which is available to anyone and can help push forward the whole flexibility market. If you build upon this model, or use it in some other way, we would be happy to see you contribute to this common goal as well.


#### Contents 

• The file [optimizer_ortools.py](optimizer_ortools.py) includes the implementation of the BESS optimization for the DA Auction, ID Auction and ID Continuous markets using OR-Tools CBC MILP solver.

• The notebook [example.ipynb](example.ipynb) demonstrates **optimization techniques** including:
  - Sequential optimization.
  - Simultaneous multi-market optimization.

• [mathematical_formulation.pdf](mathematical_formulation.pdf) is a document, which includes the mathematical formulation of the optimization problem. It is meant to accompany the code implementation. The model constraints in the code and in the mathematical formulation are numbered accordingly. 

• [LICENSE](LICENSE) is the copyright license of this work.

### Installation

```bash
pip install ortools numpy matplotlib
```

No external solver installation required - OR-Tools comes with CBC solver built-in.


#### Abbreviations

The formulation includes a number of abbreviations, below is a short list with explanations:

DAA:         Day-ahead Auction. In the German case this is the EPEX Day-ahead auction which takes place at 12:00h on the day before delivery.

IDA:         Intraday Auction. In the German case this is the EPEX Intraday auction which takes place at 15:00h on the day before delivery. 

IDC:         Intraday Continuous Market. Opens at 15:00h on the day before delivery and closes 5 minutes before delivery.

ID1:         The volume-weighted average price of all trades in a specific contract on the Intraday Continuous market.

SOC[q]:      State of Charge of a Battery at quarter q.

CHA[q]:      Charge rate of a Battery at quarter q.

DIS[q]:      Discharge rate of a Battery at quarter q.



# Macro-Driven Credit Stress Testing

**A Vasicek single-factor stress testing framework applied to a retail credit portfolio, extending a machine-learning Probability of Default (PD) model into a full CCAR-style scenario analysis.**

---

## Table of Contents

- [Project Motivation](#project-motivation)
- [How This Project Fits With the Base PD Model](#how-this-project-fits-with-the-base-pd-model)
- [Why a Direct Macro Regression Doesn't Work Here](#why-a-direct-macro-regression-doesnt-work-here)
- [The Vasicek Single-Factor Model](#the-vasicek-single-factor-model)
- [Methodology, Step by Step](#methodology-step-by-step)
- [Scenario Design](#scenario-design)
- [Key Assumptions](#key-assumptions)
- [Worked Example](#worked-example)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Limitations and What a Production Version Would Add](#limitations-and-what-a-production-version-would-add)
- [Related Work](#related-work)
- [Tech Stack](#tech-stack)

---

## Project Motivation

A Probability of Default model answers one question: *given a borrower's characteristics today, how likely are they to default?* That's useful, but it's only half of what a credit risk function actually needs to know. The other half — the question regulators, credit committees, and capital planning teams ask every year — is: **how does that risk change if the economy deteriorates?**

This is the entire premise behind regulatory stress testing regimes such as the Federal Reserve's **CCAR/DFAST** programs in the US and **IFRS 9** forward-looking expected credit loss requirements internationally. A PD model that only reflects current conditions understates risk during a downturn, because default rates are not static — they rise sharply when unemployment climbs and incomes fall, and a bank that hasn't quantified that sensitivity is under-provisioned exactly when it matters most.

This project takes an existing PD model and asks that second question directly: **if unemployment rises and GDP contracts, how much does this portfolio's expected loss increase, and which borrowers are most exposed?**

## How This Project Fits With the Base PD Model

This is the second project in a connected two-part body of work:

1. **[Credit PD Model](../credit-pd-model)** — trains an XGBoost classifier on borrower-level features (income, loan amount, credit history, home ownership, loan grade, etc.) to estimate each borrower's probability of default under current conditions.
2. **This project** — takes those same borrower-level PDs and stresses them under macroeconomic scenarios, translating individual default probabilities into portfolio-level financial impact.

The two projects share the same dataset, the same feature engineering, and the same trained model — this project simply asks what happens to that model's output when the macro environment changes. Together they represent the two halves of how a bank actually uses a PD model: point-in-time scoring, and forward-looking scenario planning.

## Why a Direct Macro Regression Doesn't Work Here

The most obvious way to link macro conditions to credit risk would be to regress a historical default rate time series against macro variables — for example, `default_rate(t) = β₀ + β₁·unemployment(t) + β₂·GDP_growth(t) + ε`. This is a legitimate and common approach, but it requires **quarterly (or monthly) vintage-level default data spanning multiple economic cycles**.

The dataset used here (`credit_risk_csv.csv`) is a **single cross-sectional snapshot** — every borrower is observed once, with no origination date and no way to reconstruct a time series of portfolio default rates. A regression-based approach simply isn't available with this data.

Rather than forcing a technique the data doesn't support, this project uses an approach designed exactly for this situation: the **Vasicek single-factor model**, which stresses individual, point-in-time PDs directly, without requiring a historical default time series at all.

## The Vasicek Single-Factor Model

The Vasicek model is not a workaround — it is the same model that underpins the **Basel II/III Internal Ratings-Based (IRB) capital formula**, the methodology banks are required to use to calculate regulatory capital against credit risk. Its core idea: each borrower's likelihood of default is driven by two components —

- an **idiosyncratic** factor specific to that borrower (their income, credit history, etc. — already captured in `PD_baseline` from the XGBoost model), and
- a **systematic** factor common to the whole economy (recession risk, unemployment, market-wide credit conditions).

The model combines these into a single closed-form transformation:

```
PD_stressed = Φ( ( Φ⁻¹(PD_baseline) − √ρ · Z ) / √(1 − ρ) )
```

Where:

| Symbol | Meaning |
|---|---|
| `PD_baseline` | The borrower's default probability under normal conditions (from the XGBoost model) |
| `ρ` (rho) | Asset correlation — how exposed this borrower's risk is to systematic/macro shocks vs. their own idiosyncratic risk |
| `Z` | The systematic macro factor for a given scenario. `Z = 0` is "normal," `Z < 0` represents an adverse economic state |
| `Φ`, `Φ⁻¹` | The standard normal cumulative distribution function, and its inverse (the probit function) |

Intuitively: the model takes a borrower's normal-conditions PD, converts it into "distance from default" in standard-deviation terms via the probit transform, shifts that distance according to how bad the economy is (`Z`) and how exposed this borrower is to macro conditions (`ρ`), and converts it back into a probability. A larger `ρ` means the borrower's fate is more tied to the broader economy; a more negative `Z` means a worse recession.

## Methodology, Step by Step

1. **Rebuild features and retrain the model** — the same feature engineering (loan-to-income ratio, income per year employed, credit history age ratio, high-risk grade flag) and XGBoost pipeline from the base PD project are reproduced here, so the stressed portfolio is scored on identical logic.
2. **Score the full portfolio** — every borrower in the dataset gets a `PD_baseline`, not just the model's test split, since the goal here is portfolio-level financial impact, not model evaluation.
3. **Design three macroeconomic scenarios** — Baseline, Adverse, and Severely Adverse, calibrated in the style of the Federal Reserve's CCAR scenario framework (see [Scenario Design](#scenario-design)).
4. **Standardize scenario shocks into a systematic factor `Z`** — each scenario's unemployment and GDP shocks are converted into standard-deviation units and combined into a single `Z` value, with an optional live pull from FRED to validate the volatility assumptions against real historical data.
5. **Apply the Vasicek transformation** — every borrower's `PD_baseline` is stressed under all three scenarios using their individual, PD-dependent asset correlation.
6. **Compute stressed Expected Loss** — `EL = PD × LGD × EAD`, aggregated to the portfolio level for each scenario.
7. **Analyze risk-band migration** — borrowers are bucketed into risk bands (Very Low through Very High risk) under each scenario, showing how many migrate upward as conditions worsen.
8. **Estimate directional capital impact** — the simplified Basel IRB risk-weight formula translates stressed PDs into an approximate risk-weighted asset (RWA) impact, connecting the stress test back to regulatory capital.

## Scenario Design

| Scenario | ΔUnemployment | GDP Growth | Narrative |
|---|---|---|---|
| **Baseline** | +0.0pp | 0.0% | Current macroeconomic conditions persist |
| **Adverse** | +2.0pp | −1.5% | A moderate recession |
| **Severely Adverse** | +5.5pp | −3.5% | A deep recession, broadly comparable in magnitude to 2008 or 2020-style shocks |

These magnitudes are calibrated in the *style* of Federal Reserve CCAR scenarios — they are illustrative and directionally realistic, not an official regulatory submission or a copy of a specific published Fed scenario.

## Key Assumptions

Every stress test rests on assumptions, and a model risk reviewer's first questions are always about these. They are stated explicitly here rather than buried in code:

| Assumption | Value | Rationale |
|---|---|---|
| Loss Given Default (LGD) | 45% | Basel Foundation-IRB benchmark for senior unsecured exposures |
| Exposure at Default (EAD) | `loan_amnt` | Term loans, fully drawn, no revolving-credit conversion factor needed |
| Asset correlation (ρ) | Basel "other retail" formula, PD-dependent | Standard regulatory formula for retail/consumer credit; ranges roughly 0.03–0.16 |
| Macro shock volatility | σ_unemployment ≈ 1.0pp/quarter, σ_GDP ≈ 2.0%/quarter | Approximate historical quarterly volatility of US macro series (validated against FRED if API key provided) |
| Weighting of unemployment vs. GDP in `Z` | 50% / 50% | A simplifying assumption, not econometrically fitted — stated explicitly as a limitation |

## Worked Example

To make the mechanism concrete: consider a borrower with a baseline PD of **10%** and an asset correlation `ρ = 0.06` (typical for a mid-range retail borrower under the Basel formula). Under the Severely Adverse scenario (`Z ≈ −2.1`, reflecting the standardized shock from +5.5pp unemployment and −3.5% GDP growth):

```
Φ⁻¹(0.10)  = −1.2816
√ρ · Z     = √0.06 × (−2.1) ≈ −0.5145
numerator  = −1.2816 − (−0.5145) = −0.7671
denominator = √(1 − 0.06) ≈ 0.9695
PD_stressed = Φ(−0.7671 / 0.9695) = Φ(−0.791) ≈ 21.4%
```

A borrower with a 10% baseline PD sees their stressed PD roughly **double to ~21%** under a severe recession scenario — this is the kind of shift the notebook computes for every borrower in the portfolio, then aggregates into portfolio-level Expected Loss.

## Results

*Populate this section after running the notebook on your dataset. Suggested items to report:*

- *Portfolio-average PD: Baseline vs. Adverse vs. Severely Adverse*
- *Total portfolio Expected Loss under each scenario, and % increase vs. Baseline*
- *Number/percentage of borrowers migrating into "High" and "Very High" risk bands under stress*
- *Directional RWA / capital impact under each scenario*
- *A screenshot of the PD distribution shift chart and the risk-band migration chart*

## Project Structure

```
macro-credit-stress-testing/
├── Macro_Credit_Stress_Testing.ipynb        # main notebook — run this
├── Macro_Credit_Stress_Testing_script.py    # plain-script version of the same code
├── requirements.txt                          # Python dependencies
└── README.md
```

## Getting Started

```bash
git clone <this-repo>
cd macro-credit-stress-testing
pip install -r requirements.txt
```

1. Place `credit_risk_csv.csv` (the same file used by the base PD model) in the project root.
2. Open `Macro_Credit_Stress_Testing.ipynb` and run it top to bottom.
3. *(Optional)* Add a free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) in the macro data cell to validate the scenario shock volatility assumptions against real historical unemployment and GDP data. The notebook falls back to the documented default assumptions if this is skipped, so it runs either way.

## Limitations and What a Production Version Would Add

This project is methodologically sound but intentionally scoped for a cross-sectional dataset. Being upfront about its boundaries is part of the exercise:

- **Scenario-calibrated, not econometrically fitted.** The systematic factor `Z` is derived from standardized scenario shocks with an assumed 50/50 unemployment/GDP weighting, not estimated from a fitted macro-default regression. A production model would use panel/vintage-level default data to estimate this relationship directly (e.g., a Merton-type default-intensity model fitted across multiple economic cycles).
- **LGD held constant.** In practice, Loss Given Default rises during downturns as collateral values fall — a more advanced version would model LGD as macro-sensitive as well, not just PD.
- **Illustrative scenario magnitudes.** The Adverse/Severely Adverse shocks are calibrated to be directionally realistic (in the style of Fed CCAR scenarios) but are not official published regulatory scenario values.
- **Simplified capital formula.** The RWA calculation omits the Basel maturity adjustment and other refinements present in the full regulatory formula — it's included here to show directional capital sensitivity, not to produce a regulatory-grade capital number.

## Related Work

- **[Credit PD Model](../credit-pd-model)** — the base XGBoost model this project extends.
- **Model Risk Validation** *(planned)* — a follow-up project applying population stability index (PSI), characteristic stability index (CSI), and AUC/KS drift monitoring to the same PD model, completing the full model lifecycle: **build → stress → validate**.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · SciPy · Matplotlib · Seaborn · (optional) FRED API

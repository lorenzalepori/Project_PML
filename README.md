# COVID-19 conterfactual analysis on vaccination using a Conditional Diffusion Model
The aim of this project is to build a conditional diffusion model to forecast the evolution of the COVID-19 pandemic in Italy and explore conterfactual scenarios under different vaccination and mobility policies.
The model learns the joint distribution of cases and deaths conditioned on mobility and vaccines over a three years timeframe.
The counterfactuals are generated under alternative scenarios to prove the effectiveness of the vaccines.
**Reference paper:** Sacco C. et al., *"Estimating averted COVID-19 cases, hospitalisations, ICU admissions and deaths by COVID-19 vaccination, Italy, January–September 2021"*, Eurosurveillance, 2021. [DOI: 10.2807/1560-7917.ES.2021.26.47.2101001](https://doi.org/10.2807/1560-7917.ES.2021.26.47.2101001)

## Project Structure
```
Project_PML/
│
├── Data/
│   ├──data_preparation.py   # Data download and preprocessing  
│   ├── raw_data/            # Downloaded raw datasets
│   └── processed_data/      # Processed panel and norm stats 
│
├── Model/
│   ├── Training_model.py     # DDPM model definition and training
│   ├── Analysis.py           # Counterfactual sampling and plots
│   └── results/              # Output plots and tables 
│
└── README.md
```

## Data sources
| Source | Description |
|--------|-------------|
| [PCM-DPC GitHub](https://github.com/pcm-dpc/COVID-19) | Daily COVID-19 cases and deaths divided by Italian region |
| [Italian Government Open Data](https://github.com/italia/covid19-opendata-vaccini) | Daily vaccine administrations by region (after vaccines became available) |
| [Google Mobility Reports](https://www.google.com/covid19/mobility/) | Regional mobility changes from baseline |
| [Eurostat](https://ec.europa.eu/eurostat) | Regional population data |
 
**Time range:** 01/03/2020 – 31/12/2022  
**Regions:** 20 Italian regions, autonomus provinces aggregated into regions

The githubs are both verified and eurostat is an official website of the EU, therefore the data were considered reliable.
 
---
 
## Features

### Past context (60 days)
| Feature | Description |
|---------|-------------|
| `cases` | Daily new cases per 100k (log-transformed, z-scored) |
| `deaths` | Daily deaths per 100k (log-transformed, z-scored) |
| `vaccine_elderly` | Cumulative vaccine coverage for 60+ (lagged 21 days, z-scored) |
| `vaccine_young` | Cumulative vaccine coverage for under 60 (lagged 21 days, z-scored) |
| `mobility` | Average mobility change from pre-pandemic baseline (z-scored) |
| `season_sin` | Sine component of day-of-year seasonality |
| `season_cos` | Cosine component of day-of-year seasonality |
| `pop_log` | Log of regional population |
| `region_id` | Regional identifier (embedded) |
 
### Future covariates (30 days)
`vaccine_elderly`, `vaccine_young`, `mobility`, `season_sin`, `season_cos`
 
### Targets (30 days)
`cases`, `deaths`

On the vaccine coverage was imposed a 21 days lag to reflect the delay between administration and immunity.

---

 ## Method
 
The model is a **Conditional DDPM** learing:
 
```
p(cases, deaths | vaccine, mobility, region, season)
```
 
- **Past context (60 days):** cases, deaths, vaccine coverage (lagged 21 days), mobility, seasonality, population
- **Future covariates (30 days):** vaccine coverage, mobility, seasonality
- **Targets (30 days):** cases, deaths

The population was not considered in the future covariates as it's assumed constant, or at least not significantly varying daily; it was also not a target since the mortality of COVID-19 was not high enough to change it drastically over three years.

Since the COVID cycle lasts approximatively 4 to 8 weeks from infection to (possible) death, a 60-day timeframe allows the model to capture full cycles, spot weekly patterns and tendencies over the middle period. The 30-day forecast timeframe follows epidemiological literature standards: it's long enough to inform policy decisions and short enough to avoid having to take into account different variations and/or waves. The 21-day lag on vaccine coverage reflects the biological delay between dose administration and protective immunity.

Counterfactual scenarios are generated via an **autoregressive rollout** of 3 consecutive 30-day windows (90 days total). In each window, only the future covariates are modified according to the scenario (e.g. vaccine coverage set to zero). The past context of each subsequent window is built from the generated output of the previous one, allowing the effect of the modified policy to propagate through time. By window 3, the model sees a past in which vaccines have been absent for 60 days, capturing the cumulative impact of the intervention.

### Scenarios
 
| Scenario | Vaccine | Mobility | Interpretation |
|----------|---------|----------|----------------|
| `no_intervention` | 0.0 | 1.0 | No vaccination, real mobility restrictions |
| `no_vax_restrictions` | 0.0 | 0.6 | No vaccination, mobility restrictions +40% (moderate lockdown) |
| `no_restrictions_vax` | 0.2 | 1.0 | vaccination -80%, real mobility restrictions |

The model doesn't simulate the scenario with higher vaccine coverage, it uses the observed reality
as upper bound and only investigates scenatios with less coverage and different mobility restrictions.
Furthermore, the model doesn't simulate the scenario with the mobilty pre-covid because the data were unavailable,
it reamins safe to assume that no intervention with even higher mobility would result in even higher number of deaths.

---

## Autoregressive Rollout
 
To capture the cumulative effect of vaccination over time, the analysis uses a **3-window autoregressive rollout** (90 days total):
 
1. **Window 1 (days 1–30):** Real observed past (60 days) → generate 30 days under scenario covariates
2. **Window 2 (days 31–60):** Last 30 real days + 30 generated days → generate next 30 days
3. **Window 3 (days 61–90):** 30 generated days + 30 generated days → generate next 30 days
Between windows, the new past context is built using:
- `cases`, `deaths` → median of generated samples from previous window
- `vaccine`, `mobility` → real panel values, scaled by scenario factor
- `season`, `pop_log`, `region_id` → always from real panel
This allows the effect of missing vaccination to propagate through time: by window 3, the model sees a past in which vaccines have been absent for 60 days.
 
---
 
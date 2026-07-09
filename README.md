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
|      Feature      |                             Description                             |
|-------------------|---------------------------------------------------------------------|
|      `cases`      |        Daily new cases per 100k (log-transformed, z-scored)         |
|      `deaths`     |          Daily deaths per 100k (log-transformed, z-scored)          |
| `vaccine_elderly` |    Cumulative vaccine coverage for 60+ (lagged 21 days, z-scored)   |
|  `vaccine_young`  | Cumulative vaccine coverage for under 60 (lagged 21 days, z-scored) |
|     `mobility`    |    Average mobility change from pre-pandemic baseline (z-scored)    |
|    `season_sin`   |             Sine component of day-of-year seasonality               |
|    `season_cos`   |            Cosine component of day-of-year seasonality              |
|     `pop_log`     |                     Log of regional population                      | 
|    `region_id`    |                   Regional identifier (embedded)                    |
 
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
 
|        Scenario       |  Vaccine  | Mobility |                        Interpretation                          |
|-----------------------|-----------|----------|----------------------------------------------------------------|
| `no_intervention`     |    0.0    |    1.0   |           No vaccination, real mobility restrictions           |
| `no_vax_restrictions` |    0.0    |    0.6   | No vaccination, mobility restrictions +40% (moderate lockdown) |
| `no_restrictions_vax` |    0.2    |    1.0   |          vaccination -80%, real mobility restrictions          |

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

## data_preparation.py

The script `data_preparation.py` builds a regional daily panel dataset for Italy using COVID-19, vaccination, mobility, and population data. Raw are cleaned by standardizing region names and selecting the period from January 1, 2021 to December 31, 2022.

COVID-19 data are processed to obtain daily cases and deaths per region. Vaccination data are split into elderly and young population groups, converted into cumulative per-capita indicators, and shifted by 21 days to account for delayed vaccine effects. Mobility data are aggregated into a single mobility index, while population data are used to convert cases and deaths into rates per 100,000 inhabitants and to create a logarithmic population-size feature.

The final dataset includes normalized regional time series for cases, deaths, vaccination indicators, mobility, seasonality features, population size, and region identifiers. The script saves the processed panel dataset as `dataset_panel.csv`, the cleaned population table as `population_clean.csv`, and the normalization statistics as `norm_stats.csv`.

The script also splits the processed data into `train.csv`, `val.csv`, and `test.csv`, where 60 days of input data are used to predict the following 30 days of cases and deaths.

## Training_model.py

The `Training_model.py` script trains a conditional diffusion model to generate 30-day future trajectories of COVID-19 cases and deaths. The model uses a 60-day historical context for each Italian region, including epidemic variables, vaccination coverage, mobility, seasonality, population information, and region identity.

The input data are loaded from `dataset_panel.csv`. The dataset is transformed into temporal windows, where each sample contains 60 days of past observations, 30 days of future covariates, and 30 days of target values. The training uses data from January to September 2021, validation uses October to December 2021, and testing uses January to March 2022.

The model is a conditional DDPM denoiser. During training, Gaussian noise is added to the true 30-day future sequence of cases and deaths. The neural network then learns to recover the added noise using the previous 60 days of epidemic information, including cases, deaths, vaccination coverage, mobility, seasonality, population features, and the region identifier. It also uses the known future variables for the next 30 days, such as vaccination levels, mobility, and seasonal components, together with a numerical indicator of the current noise level.

The training loop minimizes the mean squared error between the true noise and the predicted noise. An exponential moving average of the model weights is also maintained to improve stability. The best model according to validation loss is saved as `diffusion_ckpt.pt` inside the processed data folder.

## Analysis.py

The `Analysis.py` script uses the pre-trained conditional diffusion model to simulate counterfactual epidemic trajectories over a 90-day horizon. The prediction is performed autoregressively in three consecutive 30-day windows.

The model generates multiple stochastic samples of future cases and deaths for each Italian region under different intervention scenarios. Each scenario modifies selected future covariates, mainly vaccination coverage and mobility, while keeping the remaining information from the observed dataset.

The implemented scenarios are:

- `real_case`: observed vaccination and mobility levels
- `no_intervention`: no vaccination, observed mobility
- `no_vax_restrictions`: no vaccination and reduced mobility
- `no_restrictions_vax`: reduced vaccination and observed mobility

For each scenario, the generated normalized predictions are transformed back into absolute daily counts using the stored normalization statistics and regional population. The script then compares the simulated `real_case` trajectory with the observed data, computes calibration metrics such as MAE, and summarizes the total cases, deaths, peak cases, and differences with respect to the real-case baseline.

The script saves two CSV files: `mae_by_region.csv`, containing the regional MAE values, and `scenario_results.csv`, containing the summary statistics for each counterfactual scenario.


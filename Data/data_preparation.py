import pandas as pd
import os
import numpy as np
import requests
import re
import io
import zipfile
import shutil
import eurostat

def missing_download(file_path, download_func):
    if not os.path.exists(file_path):
        download_func()

directory = os.path.dirname(__file__)
raw_dir = os.path.join(directory, "raw_data")
processed_dir = os.path.join(directory, "processed_data")
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

region_names = {
    "ITC1": "Piemonte",
    "ITC2": "Valle d'Aosta",
    "ITC3": "Liguria",
    "ITC4": "Lombardia",
    "ITH1": "Trentino-Alto Adige",
    "ITH2": "Trentino-Alto Adige",
    "ITH3": "Veneto",
    "ITH4": "Friuli-Venezia Giulia",
    "ITH5": "Emilia-Romagna",
    "ITI1": "Toscana",
    "ITI2": "Umbria",
    "ITI3": "Marche",
    "ITI4": "Lazio",
    "ITF1": "Abruzzo",
    "ITF2": "Molise",
    "ITF3": "Campania",
    "ITF4": "Puglia",
    "ITF5": "Basilicata",
    "ITF6": "Calabria",
    "ITG1": "Sicilia",
    "ITG2": "Sardegna"
}

# Download of the COVID-19 data
covid_url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
c_path = os.path.join(raw_dir, "covid_dataset.csv")
if not os.path.exists(c_path):
    covid_data = pd.read_csv(covid_url)
    covid_data.to_csv(c_path, index=False)

# Download of the population data
p_path = os.path.join(raw_dir, "population_dataset.csv")
if not os.path.exists(p_path):

  df = eurostat.get_data_df('demo_r_pjanaggr3')
  population_data = df[df['geo\\TIME_PERIOD'].str.startswith('IT')& 
                    (df['geo\\TIME_PERIOD'].str.len() == 4)&
                    (df['age'] == 'TOTAL')] 
  population_data.to_csv(p_path, index=False)


# Download of the vaccination data
vaccines_url = "https://raw.githubusercontent.com/italia/covid19-opendata-vaccini/master/dati/somministrazioni-vaccini-summary-latest.csv"
v_path = os.path.join(raw_dir, "vaccines_dataset.csv")
if not os.path.exists(v_path):
    vaccines_data = pd.read_csv(vaccines_url)
    vaccines_data.to_csv(v_path, index=False)
 
# Download of the mobility data
m_path = os.path.join(raw_dir, "mobility_dataset.csv")
if not os.path.exists(m_path):
    zip_url = "https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip"
    r = requests.get(zip_url)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    it_files = [n for n in z.namelist()
                if "_IT_Region_Mobility_Report.csv" in n and n[:4] in ("2021", "2022")]
    frames = [pd.read_csv(z.open(n)) for n in it_files]
    mobility_data = pd.concat(frames, ignore_index=True)
    mobility_data.to_csv(m_path, index=False)

# Population data cleaning
pop_raw = pd.read_csv(p_path)
pop_raw = pop_raw[(pop_raw["sex"] == "T") & (pop_raw["age"] == "TOTAL")]
pop_clean = pop_raw[["geo\\TIME_PERIOD", "2021"]].rename(
    columns={"geo\\TIME_PERIOD": "nuts2", "2021": "population"}
)
pop_clean["region"] = pop_clean["nuts2"].map(region_names)
unmapped = pop_clean[pop_clean["region"].isna()]["nuts2"].unique()
if len(unmapped) > 0:
    print("⚠️ Codici NUTS2 non mappati:", unmapped)
pop_clean = pop_clean.dropna(subset=["region"])
pop_clean = pop_clean.groupby("region", as_index=False)["population"].sum()
pop_clean.to_csv(p_path, index=False)


# Time range for the analysis
START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2022-12-31")

# Standardization of the region names
REGIONS = ["Abruzzo", "Basilicata", "Calabria", "Campania", "Emilia-Romagna",
           "Friuli-Venezia Giulia", "Lazio", "Liguria", "Lombardia", "Marche",
           "Molise", "Piemonte", "Puglia", "Sardegna", "Sicilia", "Toscana",
           "Trentino-Alto Adige", "Umbria", "Valle d'Aosta", "Veneto"]
COVID_MAP = {"Friuli Venezia Giulia": "Friuli-Venezia Giulia",
             "P.A. Bolzano": "Trentino-Alto Adige",
             "P.A. Trento": "Trentino-Alto Adige"}
VAX_MAP = {"Provincia Autonoma Bolzano / Bozen": "Trentino-Alto Adige",
           "Provincia Autonoma Trento": "Trentino-Alto Adige",
           "Valle d'Aosta / Vallée d'Aoste": "Valle d'Aosta"}
GOOG_MAP = {"Lombardy": "Lombardia", "Piedmont": "Piemonte", "Apulia": "Puglia",
            "Sardinia": "Sardegna", "Sicily": "Sicilia", "Tuscany": "Toscana",
            "Aosta": "Valle d'Aosta", "Aosta Valley": "Valle d'Aosta",
            "Trentino-South Tyrol": "Trentino-Alto Adige"}

def report_unmapped(name, regions):
    extra = sorted(set(regions) - set(REGIONS))
    if extra:
        print(f"Warning: {name} has unmapped regions: {extra}")

# Covid data cleaning
covid = pd.read_csv(os.path.join(raw_dir, "covid_dataset.csv"),
                    usecols=["data", "denominazione_regione", "nuovi_positivi", "deceduti"])
covid["data"] = pd.to_datetime(covid["data"]).dt.normalize()
covid = covid.sort_values(["denominazione_regione", "data"])
covid["deaths"] = covid.groupby("denominazione_regione")["deceduti"].diff()
covid["cases"] = covid["nuovi_positivi"]
covid["region"] = covid["denominazione_regione"].map(COVID_MAP).fillna(covid["denominazione_regione"])
report_unmapped("COVID", covid["region"])
covid = covid[(covid["data"] >= START) & (covid["data"] <= END)]
covid_agg = covid.groupby(["region", "data"], as_index=False).agg(
    cases=("cases", "sum"), deaths=("deaths", "sum"))

# Vaccine data cleaning
vacc = pd.read_csv(os.path.join(raw_dir, "vaccines_dataset.csv"),
                   usecols=["data", "reg", "totale"])
vacc["data"] = pd.to_datetime(vacc["data"]).dt.normalize()
vacc["region"] = vacc["reg"].map(VAX_MAP).fillna(vacc["reg"])
report_unmapped("Vaccini", vacc["region"])
vacc = vacc[(vacc["data"] >= START) & (vacc["data"] <= END)]
# Somma giornaliera senza calcolare subito il cumulativo
vacc_agg = vacc.groupby(["region", "data"], as_index=False).agg(vaccine=("totale", "sum"))

# Mobility data cleaning
MOB_COLS = ["retail_and_recreation_percent_change_from_baseline",
            "grocery_and_pharmacy_percent_change_from_baseline",
            "transit_stations_percent_change_from_baseline",
            "workplaces_percent_change_from_baseline"]
mob = pd.read_csv(os.path.join(raw_dir, "mobility_dataset.csv"))
mob["data"] = pd.to_datetime(mob["date"]).dt.normalize()
mob = mob[mob["sub_region_1"].notna() & mob["sub_region_2"].isna()].copy()
mob["mobility"] = mob[MOB_COLS].mean(axis=1)
mob["region"] = mob["sub_region_1"].map(GOOG_MAP).fillna(mob["sub_region_1"])
report_unmapped("Mobilita'", mob["region"])
mob = mob[["region", "data", "mobility"]]
# Add missing mobility data for the last year by using the previous year's data
mob = mob[mob["data"] <= END]
last_real = mob["data"].max()
gap_start = last_real + pd.Timedelta(days=1)
proxy = mob[(mob["data"] >= gap_start - pd.DateOffset(years=1)) &
            (mob["data"] <= END - pd.DateOffset(years=1))].copy()
proxy["data"] = proxy["data"] + pd.DateOffset(years=1)
mob = pd.concat([mob, proxy], ignore_index=True)
mob = mob[(mob["data"] >= START) & (mob["data"] <= END)]
mob_agg = mob.groupby(["region", "data"], as_index=False).agg(mobility=("mobility", "mean"))

# Merge
cal = pd.MultiIndex.from_product(
    [REGIONS, pd.date_range(START, END, freq="D")],
    names=["region", "data"]).to_frame(index=False)

panel = (cal.merge(covid_agg, on=["region", "data"], how="left")
            .merge(vacc_agg, on=["region", "data"], how="left")
            .merge(mob_agg, on=["region", "data"], how="left"))

# Manage impossible values (negative cases, deaths or vaccines)
for c in ["cases", "deaths", "vaccine"]:
    panel[c] = panel[c].clip(lower=0)
panel["cases"] = panel["cases"].fillna(0)
panel["deaths"] = panel["deaths"].fillna(0)
# Riempie giorni mancanti e poi calcola il cumulativo corretto
panel["vaccine"] = panel.groupby("region")["vaccine"].ffill().fillna(0)
panel["vaccine"] = panel.groupby("region")["vaccine"].cumsum()
panel["mobility"] = (panel.groupby("region")["mobility"]
                     .transform(lambda s: s.interpolate(limit_direction="both")))

# Merge with population data
population_dataset = pd.read_csv(p_path)   
pop = population_dataset[["region", "population"]].copy()
panel = panel.merge(pop, on="region", how="left")
panel["vaccine"] = panel["vaccine"] / panel["population"]
panel["pop_log"] = np.log(panel["population"])

# Weekly average on cases, deaths and mobility (sui valori assoluti)
for c in ["cases", "deaths", "mobility"]:
    panel[c] = (panel.groupby("region")[c]
                .transform(lambda s: s.rolling(7, center=True, min_periods=1).mean()))

# Trasformation for 100.000 inhabitants and log
panel["cases_rate"]  = 100000 * panel["cases"] / panel["population"]
panel["deaths_rate"] = 100000 * panel["deaths"] / panel["population"]

panel["cases_rate"]  = np.log1p(panel["cases_rate"])
panel["deaths_rate"] = np.log1p(panel["deaths_rate"])

panel["cases"]  = panel["cases_rate"]
panel["deaths"] = panel["deaths_rate"]

# Remove temporary columns
panel.drop(columns=["cases_rate", "deaths_rate"], inplace=True)

# Z-score per region
NORM_VARS = ["cases", "deaths", "vaccine", "mobility"]
stats = []
for region, g in panel.groupby("region"):
    for v in NORM_VARS:
        mu = g[v].mean()
        sigma = g[v].std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            sigma = 1.0
        panel.loc[g.index, v] = (g[v] - mu) / sigma
        stats.append({"region": region, "variable": v, "mu": mu, "sigma": sigma})
norm_stats = pd.DataFrame(stats)
norm_stats.to_csv(os.path.join(processed_dir, "norm_stats.csv"), index=False)

# Seasonality features
doy = panel["data"].dt.dayofyear
panel["season_sin"] = np.sin(2 * np.pi * doy / 365)
panel["season_cos"] = np.cos(2 * np.pi * doy / 365)

# Region map
region_id_map = {r: i for i, r in enumerate(REGIONS)}
panel["region_id"] = panel["region"].map(region_id_map)

# Panel
panel = panel.rename(columns={"data": "date"})
# Corretto l'elenco delle colonne: season_cos non duplicato
COLS = ["date", "region", "cases", "deaths", "vaccine", "mobility",
        "season_sin", "season_cos", "pop_log", "region_id"]
panel = panel[COLS].sort_values(["region", "date"]).reset_index(drop=True)

# Temporal features: 60 days input and 30 days output
FEATURES = ["cases", "deaths", "vaccine", "mobility", "season_sin", "season_cos", "pop_log", "region_id"]
TARGETS = ["cases", "deaths"]
INPUT_LEN, OUTPUT_LEN = 60, 30

def make_windows(df, period_start, period_end):
    """Finestre i cui 30 giorni di target cadono interamente in [period_start, period_end].
    L'input (60 gg) puo' attingere allo storico precedente: nessun leakage dal futuro."""
    Xs, Ys, reg, rid, t0 = [], [], [], [], []
    for region, g in df.groupby("region"):
        g = g.sort_values("date").reset_index(drop=True)
        feat = g[FEATURES].to_numpy(dtype="float32")
        targ = g[TARGETS].to_numpy(dtype="float32")
        dates = g["date"].to_numpy()
        for t in range(INPUT_LEN, len(g) - OUTPUT_LEN + 1):
            tgt_start, tgt_end = g["date"].iloc[t], g["date"].iloc[t + OUTPUT_LEN - 1]
            if tgt_start >= period_start and tgt_end <= period_end:
                Xs.append(feat[t - INPUT_LEN:t])
                Ys.append(targ[t:t + OUTPUT_LEN])
                reg.append(region)
                rid.append(int(g["region_id"].iloc[0]))
                t0.append(dates[t])
    X = np.stack(Xs) if Xs else np.empty((0, INPUT_LEN, len(FEATURES)), "float32")
    Y = np.stack(Ys) if Ys else np.empty((0, OUTPUT_LEN, len(TARGETS)), "float32")
    return X, Y, np.array(reg), np.array(rid), np.array(t0, dtype="datetime64[ns]")

# Temporal split and save
SPLITS = {
    "train": (START, pd.Timestamp("2022-06-30")),
    "val":   (pd.Timestamp("2022-07-01"), pd.Timestamp("2022-09-30")),
    "test":  (pd.Timestamp("2022-10-01"), END),
}

panel.to_csv(os.path.join(processed_dir, "dataset_panel.csv"), index=False)

for name, (s, e) in SPLITS.items():
    fetta = panel[(panel["date"] >= s) & (panel["date"] <= e)]
    fetta.to_csv(os.path.join(processed_dir, f"{name}.csv"), index=False)
    X, Y, reg, rid, t0 = make_windows(panel, s, e)
    np.savez_compressed(os.path.join(processed_dir, f"{name}_windows.npz"),
                        X=X, Y=Y, region=reg, region_id=rid, target_start=t0)
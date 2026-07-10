import pandas as pd
import os
import numpy as np
import requests
import io
import zipfile
import eurostat

directory = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(directory, "raw_data")
processed_dir = os.path.join(directory, "processed_data")
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

region_names = {
    "ITC1": "Piemonte", "ITC2": "Valle d'Aosta", "ITC3": "Liguria",
    "ITC4": "Lombardia", "ITH1": "Trentino-Alto Adige", "ITH2": "Trentino-Alto Adige",
    "ITH3": "Veneto", "ITH4": "Friuli-Venezia Giulia", "ITH5": "Emilia-Romagna",
    "ITI1": "Toscana", "ITI2": "Umbria", "ITI3": "Marche", "ITI4": "Lazio",
    "ITF1": "Abruzzo", "ITF2": "Molise", "ITF3": "Campania", "ITF4": "Puglia",
    "ITF5": "Basilicata", "ITF6": "Calabria", "ITG1": "Sicilia", "ITG2": "Sardegna"
}

# --- Downloads ---
covid_url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
c_path = os.path.join(raw_dir, "covid_dataset.csv")
if not os.path.exists(c_path):
    pd.read_csv(covid_url).to_csv(c_path, index=False)

# Population total + over65 
p_path   = os.path.join(raw_dir, "population_dataset.csv")
age_path = os.path.join(raw_dir, "population_age_dataset.csv")
if not os.path.exists(p_path) or not os.path.exists(age_path):
    df_full = eurostat.get_data_df('demo_r_pjanaggr3')
    it_mask = (df_full['geo\\TIME_PERIOD'].str.startswith('IT') &
               (df_full['geo\\TIME_PERIOD'].str.len() == 4) &
               (df_full['sex'] == 'T'))
    year_cols = [c for c in df_full.columns if c.isdigit() and int(c) >= 2019]
    keep_cols = ['freq', 'unit', 'sex', 'age', 'geo\\TIME_PERIOD'] + year_cols
    # Total population
    df_full[it_mask & (df_full['age'] == 'TOTAL')][keep_cols].to_csv(p_path, index=False)
    # Over-65 population
    df_full[it_mask & (df_full['age'] == 'Y_GE65')][keep_cols].to_csv(age_path, index=False)

# Vaccination by age group
vacc_age_url = "https://raw.githubusercontent.com/italia/covid19-opendata-vaccini/master/dati/somministrazioni-vaccini-latest.csv"
va_path = os.path.join(raw_dir, "vaccines_age_dataset.csv")
if not os.path.exists(va_path):
    pd.read_csv(vacc_age_url).to_csv(va_path, index=False)

m_path = os.path.join(raw_dir, "mobility_dataset.csv")
if not os.path.exists(m_path):
    zip_url = "https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip"
    r = requests.get(zip_url)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    it_files = [n for n in z.namelist()
                if "_IT_Region_Mobility_Report.csv" in n and n[:4] in ("2020", "2021", "2022")]
    pd.concat([pd.read_csv(z.open(n)) for n in it_files], ignore_index=True).to_csv(m_path, index=False)

# --- Population cleaning ---
pop_raw = pd.read_csv(p_path)
if {"region", "population"}.issubset(pop_raw.columns):
    pop_clean = pop_raw[["region", "population"]].copy()
else:
    pop_raw = pop_raw[(pop_raw["sex"] == "T") & (pop_raw["age"] == "TOTAL")]
    pop_clean = pop_raw[["geo\\TIME_PERIOD", "2021"]].rename(
        columns={"geo\\TIME_PERIOD": "nuts2", "2021": "population"})
    pop_clean["region"] = pop_clean["nuts2"].map(region_names)
    pop_clean = pop_clean.dropna(subset=["region"])
    pop_clean = pop_clean.groupby("region", as_index=False)["population"].sum()

pc_path = os.path.join(processed_dir, "population_clean.csv")
pop_clean.to_csv(pc_path, index=False)

# --- Over-65 population per region ---
age_raw = pd.read_csv(age_path)
age_raw["region"] = age_raw["geo\\TIME_PERIOD"].map(region_names)
age_raw = age_raw.dropna(subset=["region"])
pop_over65 = (age_raw.groupby("region")["2021"]
              .first()
              .reset_index()
              .rename(columns={"2021": "pop_over65"}))
pop_over65_map = pop_over65.set_index("region")["pop_over65"].to_dict()

START = pd.Timestamp("2021-01-01")
END   = pd.Timestamp("2022-12-31")

REGIONS = ["Abruzzo", "Basilicata", "Calabria", "Campania", "Emilia-Romagna",
           "Friuli-Venezia Giulia", "Lazio", "Liguria", "Lombardia", "Marche",
           "Molise", "Piemonte", "Puglia", "Sardegna", "Sicilia", "Toscana",
           "Trentino-Alto Adige", "Umbria", "Valle d'Aosta", "Veneto"]
COVID_MAP = {"Friuli Venezia Giulia": "Friuli-Venezia Giulia",
             "P.A. Bolzano": "Trentino-Alto Adige", "P.A. Trento": "Trentino-Alto Adige"}
VAX_MAP   = {"Provincia Autonoma Bolzano / Bozen": "Trentino-Alto Adige",
             "Provincia Autonoma Trento": "Trentino-Alto Adige",
             "Valle d'Aosta / Vallée d'Aoste": "Valle d'Aosta"}
GOOG_MAP  = {"Lombardy": "Lombardia", "Piedmont": "Piemonte", "Apulia": "Puglia",
             "Sardinia": "Sardegna", "Sicily": "Sicilia", "Tuscany": "Toscana",
             "Aosta": "Valle d'Aosta", "Aosta Valley": "Valle d'Aosta",
             "Trentino-South Tyrol": "Trentino-Alto Adige"}

# Age groups considered elderly
ELDERLY_AGES = ["60-69", "70-79", "80-89", "80+", "90+"]

def report_unmapped(name, regions):
    extra = sorted(set(regions) - set(REGIONS))
    if extra:
        print(f"Warning: {name} has unmapped regions: {extra}")

# --- COVID cleaning ---
covid = pd.read_csv(os.path.join(raw_dir, "covid_dataset.csv"),
                    usecols=["data", "denominazione_regione", "nuovi_positivi", "deceduti"])
covid["data"]   = pd.to_datetime(covid["data"]).dt.normalize()
covid           = covid.sort_values(["denominazione_regione", "data"])
covid["deaths"] = covid.groupby("denominazione_regione")["deceduti"].diff()
covid["cases"]  = covid["nuovi_positivi"]
covid["region"] = covid["denominazione_regione"].map(COVID_MAP).fillna(covid["denominazione_regione"])
report_unmapped("COVID", covid["region"])
covid     = covid[(covid["data"] >= START) & (covid["data"] <= END)]
covid_agg = covid.groupby(["region", "data"], as_index=False).agg(
    cases=("cases", "sum"), deaths=("deaths", "sum"))

# --- Vaccine by age group cleaning ---
vacc_age = pd.read_csv(va_path, usecols=["data","eta", "m", "f", "reg"])
vacc_age = vacc_age.rename(columns={"reg": "region"})
vacc_age["totale"] = vacc_age["m"] + vacc_age["f"]
vacc_age["data"]   = pd.to_datetime(vacc_age["data"]).dt.normalize()
vacc_age["region"] = vacc_age["region"].map(VAX_MAP).fillna(vacc_age["region"])

report_unmapped("Vaccini age", vacc_age["region"])
vacc_age = vacc_age[(vacc_age["data"] >= START) & (vacc_age["data"] <= END)]

# Elderly doses (60+)
vacc_elderly = (vacc_age[vacc_age["eta"].isin(ELDERLY_AGES)]
                .groupby(["region", "data"], as_index=False)
                .agg(vaccine_elderly=("totale", "sum")))

# Young doses
vacc_young = (vacc_age[~vacc_age["eta"].isin(ELDERLY_AGES)]
              .groupby(["region", "data"], as_index=False)
              .agg(vaccine_young=("totale", "sum")))

# --- Mobility cleaning ---
MOB_COLS = ["retail_and_recreation_percent_change_from_baseline",
            "grocery_and_pharmacy_percent_change_from_baseline",
            "transit_stations_percent_change_from_baseline",
            "workplaces_percent_change_from_baseline"]
mob = pd.read_csv(os.path.join(raw_dir, "mobility_dataset.csv"))
mob["data"]     = pd.to_datetime(mob["date"]).dt.normalize()
mob             = mob[mob["sub_region_1"].notna() & mob["sub_region_2"].isna()].copy()
mob["mobility"] = mob[MOB_COLS].mean(axis=1)
mob["region"]   = mob["sub_region_1"].map(GOOG_MAP).fillna(mob["sub_region_1"])
report_unmapped("Mobilita'", mob["region"])
mob = mob[["region", "data", "mobility"]]
mob = mob[mob["data"] <= END]
last_real = mob["data"].max()
gap_start = last_real + pd.Timedelta(days=1)
proxy = mob[(mob["data"] >= gap_start - pd.DateOffset(years=1)) &
            (mob["data"] <= END - pd.DateOffset(years=1))].copy()
proxy["data"] = proxy["data"] + pd.DateOffset(years=1)
mob     = pd.concat([mob, proxy], ignore_index=True)
mob     = mob[(mob["data"] >= START) & (mob["data"] <= END)]
mob_agg = mob.groupby(["region", "data"], as_index=False).agg(mobility=("mobility", "mean"))

# --- Merge ---
cal = pd.MultiIndex.from_product(
    [REGIONS, pd.date_range(START, END, freq="D")],
    names=["region", "data"]).to_frame(index=False)

panel = (cal.merge(covid_agg,    on=["region", "data"], how="left")
            .merge(vacc_elderly, on=["region", "data"], how="left")
            .merge(vacc_young,   on=["region", "data"], how="left")
            .merge(mob_agg,      on=["region", "data"], how="left"))

# 2019 = pre-pandemic baseline
panel["cases"]          = panel["cases"].fillna(0)
panel["deaths"]         = panel["deaths"].fillna(0)
panel["vaccine_elderly"] = panel.groupby("region")["vaccine_elderly"].ffill().fillna(0)
panel["vaccine_young"]   = panel.groupby("region")["vaccine_young"].ffill().fillna(0)
panel["mobility"] = panel["mobility"].fillna(0.0)

# Clip impossible values
for c in ["cases", "deaths", "vaccine_elderly", "vaccine_young"]:
    panel[c] = panel[c].clip(lower=0) 

panel["mobility"] = (panel.groupby("region")["mobility"]
                     .transform(lambda s: s.interpolate(limit_direction="both")))

# Merge population
pop   = pd.read_csv(pc_path)[["region", "population"]].copy()
panel = panel.merge(pop, on="region", how="left")

# Cumulative vaccine doses per capita
panel["pop_over65"] = panel["region"].map(pop_over65_map)
panel["pop_over65"] = panel["pop_over65"].fillna(panel["population"] * 0.23)  # fallback ~23%
panel["pop_young"]  = panel["population"] - panel["pop_over65"]

panel["vaccine_elderly"] = (panel.groupby("region")["vaccine_elderly"].cumsum()
                             / panel["pop_over65"])
panel["vaccine_young"]   = (panel.groupby("region")["vaccine_young"].cumsum()
                             / panel["pop_young"])
LAG_DAYS = 21
panel["vaccine_elderly"] = panel.groupby("region")["vaccine_elderly"].shift(LAG_DAYS)
panel["vaccine_young"]   = panel.groupby("region")["vaccine_young"].shift(LAG_DAYS)
panel["vaccine_elderly"] = panel["vaccine_elderly"].fillna(0)
panel["vaccine_young"]   = panel["vaccine_young"].fillna(0)
panel["pop_log"] = np.log(panel["population"])
panel.drop(columns=["pop_over65", "pop_young"], inplace=True)

# Weekly rolling average on cases, deaths and mobility
for c in ["cases", "deaths", "mobility"]:
    panel[c] = (panel.groupby("region")[c]
                .transform(lambda s: s.rolling(7, center=True, min_periods=1).mean()))

# Log-transform rates per 100k
panel["cases"]  = np.log1p(100000 * panel["cases"]  / panel["population"])
panel["deaths"] = np.log1p(100000 * panel["deaths"] / panel["population"])

NORM_VARS = ["cases", "deaths", "vaccine_elderly", "vaccine_young", "mobility"]
stats = []

for v in NORM_VARS:
    means = panel.groupby("region")[v].transform("mean")
    stds = panel.groupby("region")[v].transform(lambda x: x.std(ddof=0))
    stds = stds.replace(0, 1.0).fillna(1.0) # previene divisioni per zero
    
    region_stats = panel.groupby("region")[v].agg(["mean", lambda x: x.std(ddof=0)]).reset_index()
    for _, row in region_stats.iterrows():
        stats.append({"region": row["region"], "variable": v, "mu": row["mean"], "sigma": row.iloc[2] if row.iloc[2] != 0 else 1.0})
        
    panel[v] = (panel[v] - means) / stds

pd.DataFrame(stats).to_csv(os.path.join(processed_dir, "norm_stats.csv"), index=False)

# Seasonality features
doy = panel["data"].dt.dayofyear
panel["season_sin"] = np.sin(2 * np.pi * doy / 365)
panel["season_cos"] = np.cos(2 * np.pi * doy / 365)

# Region ID map
region_id_map      = {r: i for i, r in enumerate(REGIONS)}
panel["region_id"] = panel["region"].map(region_id_map)

# Rename and select columns
panel = panel.rename(columns={"data": "date"})
COLS = ["date", "region", "cases", "deaths", "vaccine_elderly", "vaccine_young",
        "mobility", "season_sin", "season_cos", "pop_log", "region_id"]
panel = panel[COLS].sort_values(["region", "date"]).reset_index(drop=True)

# Normalised panel
panel.to_csv(os.path.join(processed_dir, "dataset_panel.csv"), index=False)

# Window building
FEATURES  = ["cases", "deaths", "vaccine_elderly", "vaccine_young", "mobility",
             "season_sin", "season_cos", "pop_log", "region_id"]
TARGETS   = ["cases", "deaths"]
INPUT_LEN, OUTPUT_LEN = 60, 30

def make_windows(df, period_start, period_end):
    Xs, Ys, reg, rid, t0 = [], [], [], [], []
    for region, g in df.groupby("region"):
        g    = g.sort_values("date").reset_index(drop=True)
        feat = g[FEATURES].to_numpy(dtype="float32")
        targ = g[TARGETS].to_numpy(dtype="float32")
        dates = g["date"].to_numpy()
        for t in range(INPUT_LEN, len(g) - OUTPUT_LEN + 1):
            tgt_start = g["date"].iloc[t]
            tgt_end   = g["date"].iloc[t + OUTPUT_LEN - 1]
            if tgt_start >= period_start and tgt_end <= period_end:
                Xs.append(feat[t - INPUT_LEN:t])
                Ys.append(targ[t:t + OUTPUT_LEN])
                reg.append(region)
                rid.append(int(g["region_id"].iloc[0]))
                t0.append(dates[t])
    X = np.stack(Xs) if Xs else np.empty((0, INPUT_LEN, len(FEATURES)), "float32")
    Y = np.stack(Ys) if Ys else np.empty((0, OUTPUT_LEN, len(TARGETS)),  "float32")
    return X, Y, np.array(reg), np.array(rid), np.array(t0, dtype="datetime64[ns]")

SPLITS = {
    "train": (START,                        pd.Timestamp("2021-09-30")),
    "val":   (pd.Timestamp("2021-10-01"),   pd.Timestamp("2021-12-31")),
    "test":  (pd.Timestamp("2022-01-01"),   pd.Timestamp("2022-03-31")),
}

for name, (s, e) in SPLITS.items():
    fetta = panel[(panel["date"] >= s) & (panel["date"] <= e)]
    fetta.to_csv(os.path.join(processed_dir, f"{name}.csv"), index=False)
    X, Y, r, ri, t0 = make_windows(panel, s, e)
    np.savez_compressed(os.path.join(processed_dir, f"{name}_windows.npz"),
                        X=X, Y=Y, region=r, region_id=ri, target_start=t0)
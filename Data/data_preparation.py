import pandas as pd
import os
import requests
from io import StringIO
import urllib3
import eurostat
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



directory = os.path.dirname(__file__)
raw_dir = os.path.join(directory, "raw_data")
processed_dir = os.path.join(directory, "processed_data")
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

#Download and aggregation of the West Nile data
wn_data = {
    "wn_2019": "https://raw.githubusercontent.com/fbranda/west-nile/main/2019/human-surveillance/wn-ita-regions-human-surveillance-2019.csv",
    "wn_2020": "https://raw.githubusercontent.com/fbranda/west-nile/main/2020/human-surveillance/wn-ita-regions-human-surveillance-2020.csv",
    "wn_2021": "https://raw.githubusercontent.com/fbranda/west-nile/main/2021/human-surveillance/wn-ita-regions-human-surveillance-2021.csv",
    "wn_2022": "https://raw.githubusercontent.com/fbranda/west-nile/main/2022/human-surveillance/wn-ita-regions-human-surveillance-2022.csv",
    "wn_2023": "https://raw.githubusercontent.com/fbranda/west-nile/main/2023/human-surveillance/wn-ita-regions-human-surveillance-2023.csv",
    "wn_2024": "https://raw.githubusercontent.com/fbranda/west-nile/main/2024/human-surveillance/wn-ita-regions-human-surveillance-2024.csv"
}

dataframes = []

for year, url in wn_data.items():

    df = pd.read_csv(url)
    df['year']=year
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
w_output_path = os.path.join(raw_dir, "wn_dataset.csv")
combined_df.to_csv(w_output_path, index=False)


#Download of the Influenza data
influenza_url =  "https://raw.githubusercontent.com/fbranda/influnet/main/data-aggregated/epidemiological_data/regional_cases.csv"
    
influenza_data = pd.read_csv(influenza_url)
i_output_path = os.path.join(raw_dir, "influenza_dataset.csv")
influenza_data.to_csv(i_output_path, index=False)


#Download of the COVID-19 data
covid_url = "https://covid19.infn.it/grafici/regioni.csv"

r = requests.get(covid_url, verify=False)
r.raise_for_status()

covid_data = pd.read_csv(StringIO(r.text))
c_output_path = os.path.join(raw_dir,"covid_dataset.csv")
covid_data.to_csv(c_output_path, index=False)

#Download of the population data
df = eurostat.get_data_df('demo_r_pjanaggr3')
population_data = df[df['geo\\TIME_PERIOD'].str.startswith('IT')& 
                    (df['geo\\TIME_PERIOD'].str.len() == 4)&
                    (df['age'] == 'TOTAL')] 
p_path = os.path.join(raw_dir, "population_dataset.csv")
population_data.to_csv(p_path, index=False)

#Download of the deaths data
df = eurostat.get_data_df('hlth_cd_acdr2')
deaths_data = df[df['geo\\TIME_PERIOD'].str.startswith('IT')& 
                (df['geo\\TIME_PERIOD'].str.len() == 4)&
                (df['age'] == 'TOTAL')]
d_path = os.path.join(raw_dir, "deaths_dataset.csv")
deaths_data.to_csv(d_path, index=False)

#Download of the mobility data
df = eurostat.get_data_df('tour_occ_nin2m')
mobility_data = df[df['geo\\TIME_PERIOD'].str.startswith('IT')& 
                    (df['geo\\TIME_PERIOD'].str.len() == 4)&
                    (df['c_resid'] == 'DOM') &
                    (df['unit'] == 'NR')]
m_path = os.path.join(raw_dir, "mobility_dataset.csv")
mobility_data.to_csv(m_path, index=False)
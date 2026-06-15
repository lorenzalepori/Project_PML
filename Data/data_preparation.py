import pandas as pd
import os
import requests
import re
import io
import zipfile
import eurostat

def missing_download(file_path,download_func):
    if not os.path.exists(file_path):
        download_func()
    
directory = os.path.dirname(__file__)
raw_dir = os.path.join(directory, "raw_data")
processed_dir = os.path.join(directory, "processed_data")
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

#Download of the COVID-19 data
covid_url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"

c_path = os.path.join(raw_dir,"covid_dataset.csv")
if not os.path.exists(c_path):

  r = requests.get(covid_url, verify=False)
  r.raise_for_status()

  covid_data = pd.read_csv(StringIO(r.text))
  covid_data.to_csv(c_path, index=False)

#Download of the population data
p_path = os.path.join(raw_dir, "population_dataset.csv")
if not os.path.exists(p_path):

  df = eurostat.get_data_df('demo_r_pjanaggr3')
  population_data = df[df['geo\\TIME_PERIOD'].str.startswith('IT')& 
                    (df['geo\\TIME_PERIOD'].str.len() == 4)&
                    (df['age'] == 'TOTAL')] 
  population_data.to_csv(p_path, index=False)

#Download vaccination data 
vaccini_url = "https://raw.githubusercontent.com/italia/covid19-opendata-vaccini/master/dati/somministrazioni-vaccini-summary-latest.csv"

v_path = os.path.join(raw_dir, "vaccines_dataset.csv")
if not os.path.exists(v_path):
    vaccines_data = pd.read_csv(vaccini_url)
    vaccines_data.to_csv(v_path, index=False)

#Download of the mobility data
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

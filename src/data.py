import os
import pandas as pd


DATA_DIR = "../Data/NDSSL"
OUT_DIR = "../Data/Processed"
COLUMNS = {
    "Guinea": {
        "Region": "Totals",
        "Column": "Description",
        "Date": "Date",
        "Total cases of confirmed": ["E", "I"],
        "Number of contacts followed today": ["QE", "QI"],
        "Total deaths of confirmed": "F",
        "Total number of admissions to CTE": "H"
    },
    "Liberia": {
        "Region": "National",
        "Column": "Variable",
        "Date": "Date",
        "Total confirmed cases": ["E", "I"],
        "Currently under follow-up": ["QE", "QI"],
        "Total death/s in confirmed cases": "F",
        "Total no. currently in Treatment Units": "H"
    },
    "SierraLeone": {
        "Region": "National",
        "Column": "variable",
        "Date": "date",
        "cum_confirmed": ["E", "I"],
        "contacts_followed": ["QE", "QI"],
        "death_confirmed": "F",
        "etc_cum_admission": "H"
    }
}

for country in COLUMNS.keys():
    mappings = COLUMNS.get(country)
    region = mappings.get("Region")
    descr_column = mappings.get("Column")
    dt_column = mappings.get("Date")
    col_data = [k for k, v in mappings.items() if k not in ["Region", "Column", "Date"]]
    col_df = ['-'.join(v) for k, v in mappings.items() if k not in ["Region", "Column", "Date"]]
    df = pd.DataFrame([], columns=["Country", "Date"] + col_df)

    country_path = os.path.join(DATA_DIR, country)
    for file in os.listdir(country_path):
        if not file.endswith(".csv"): continue
        df_file = pd.read_csv(os.path.join(country_path, file), sep=",", header=0)
        dt = pd.to_datetime(df_file[dt_column].iloc[0])

        col_values = []
        for col in col_data:
            val = df_file.loc[df_file[descr_column] == col, region]
            val = int(float(str(val.iloc[0]).replace(",", ""))) if len(val) != 0 and not pd.isna(val.iloc[0]) else -1
            col_values.append(val)

        df.loc[len(df.index)] = [country, dt] + col_values
    df.to_csv(os.path.join(OUT_DIR, f"{country}_raw.csv"), index=False)


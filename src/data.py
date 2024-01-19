import os
import pandas as pd


DATA_DIR = "../Data/NDSSL"
OUT_DIR = "../Data/Processed"
COLUMNS = {
    "Guinea": {
        "Region": "Totals",
        "Column": "Description",
        "Date": "Date",
        "New cases of confirmed": {"Letter": ["E", "I"], "Lag": 19},
        "Number of contacts followed today": {"Letter": ["QE", "QI"], "Lag": 0},
        "New deaths registered": {"Letter": "F", "Lag": 5},
        "New admits to CTE so far": {"Letter": "H", "Lag": 13}
    },
    "Liberia": {
        "Region": "National",
        "Column": "Variable",
        "Date": "Date",
        "New case/s (confirmed)": {"Letter": ["E", "I"], "Lag": 19},
        "Currently under follow-up": {"Letter": ["QE", "QI"], "Lag": 0},
        "Newly reported deaths": {"Letter": "F", "Lag": 5},
        "Total no. currently in Treatment Units": {"Letter": "H", "Lag": 0}
    },
    "SierraLeone": {
        "Region": "National",
        "Column": "variable",
        "Date": "date",
        "new_confirmed": {"Letter": ["E", "I"], "Lag": 19},
        "contacts_followed": {"Letter": ["QE", "QI"], "Lag": 0},
        "death_confirmed": {"Letter": "F", "Lag": 5, "Cumulated": True},
        "etc_cum_admission": {"Letter": "H", "Lag": 13}
    }
}

for country in COLUMNS.keys():
    mappings = COLUMNS.get(country)
    region = mappings.get("Region")
    descr_column = mappings.get("Column")
    dt_column = mappings.get("Date")
    col_data = [k for k, v in mappings.items() if k not in ["Region", "Column", "Date"]]
    col_df = ['-'.join(v.get("Letter")) for k, v in mappings.items() if k not in ["Region", "Column", "Date"]]
    df = pd.DataFrame([], columns=["Country", "Date"] + col_df)

    country_path = os.path.join(DATA_DIR, country)
    for file in os.listdir(country_path):
        if not file.endswith(".csv"): continue
        df_file = pd.read_csv(os.path.join(country_path, file), sep=",", header=0)
        dt = pd.to_datetime(df_file[dt_column].iloc[0])

        col_values = []
        for col in col_data:
            val = df_file.loc[df_file[descr_column] == col, region]
            val = int(float(str(val.iloc[0]).replace(",", ""))) if len(val) != 0 and not pd.isna(val.iloc[0]) else pd.NA
            col_values.append(val)

        df.loc[len(df.index)] = [country, dt] + col_values
    df = df.set_index(pd.DatetimeIndex(df["Date"]))
    df.sort_index(inplace=True, ascending=True)

    for col_name, col_map in zip(col_df, col_data):
        df[col_name].fillna(method="ffill", inplace=True)
        df[col_name].fillna(value=0, inplace=True)

        if mappings.get(col_map).get("Cumulated", False):
            df[col_name] = df[col_name] - df[col_name].shift(1)

        lag = mappings.get(col_map).get("Lag")
        df[col_name] = df[col_name].rolling(window=f"{lag + 1}D", min_periods=1).sum()

    df.to_excel(os.path.join(OUT_DIR, f"{country}.xlsx"), index=False)


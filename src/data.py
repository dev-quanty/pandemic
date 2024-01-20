import os
import pandas as pd


# Define lags
LAG_INFECTED = 19
LAG_DEATHS = 5
LAG_HOSPITAL = 13

# Define variables for country data
DATA_DIR = "../Data/NDSSL"
COLUMNS = {
    "Guinea": {
        "Region": "Totals",
        "Column": "Description",
        "Date": "Date",
        "New cases of confirmed": {"Letter": ["E", "I"], "Lag": LAG_INFECTED},
        "Number of contacts followed today": {"Letter": ["QE", "QI"], "Lag": 0},
        "New deaths registered": {"Letter": "F", "Lag": LAG_DEATHS},
        "New admits to CTE so far": {"Letter": "H", "Lag": LAG_HOSPITAL}
    },
    "Liberia": {
        "Region": "National",
        "Column": "Variable",
        "Date": "Date",
        "New case/s (confirmed)": {"Letter": ["E", "I"], "Lag": LAG_INFECTED},
        "Currently under follow-up": {"Letter": ["QE", "QI"], "Lag": 0},
        "Newly reported deaths": {"Letter": "F", "Lag": LAG_DEATHS},
        "Total no. currently in Treatment Units": {"Letter": "H", "Lag": 0}
    },
    "SierraLeone": {
        "Region": "National",
        "Column": "variable",
        "Date": "date",
        "new_confirmed": {"Letter": ["E", "I"], "Lag": LAG_INFECTED},
        "contacts_followed": {"Letter": ["QE", "QI"], "Lag": 0},
        "death_confirmed": {"Letter": "F", "Lag": LAG_DEATHS, "Cumulated": True},
        "etc_cum_admission": {"Letter": "H", "Lag": LAG_HOSPITAL}
    }
}

# Define variables for WHO Data
WHO_DIR = "../Data/WHO"
WHO_COLUMNS = ("Week", "Sr_Confirmed", "Sr_Probable", "Pd_Confirmed", "Pd_Probable")

# Other variables
OUT_DIR = "../Data/Processed"

for country in COLUMNS.keys():
    # Parse single files for each country
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

    # Use WHO Data for infection data
    df_who = pd.read_csv(os.path.join(WHO_DIR, country + ".csv"), sep=",", index_col=False, names=WHO_COLUMNS, skiprows=4)
    df_who["Date"] = df_who["Week"].apply(lambda x: x.split(" to ")[-1].split(" (")[0])
    df_who["Date"] = pd.to_datetime(df_who["Date"])
    df_who = df_who.set_index(pd.DatetimeIndex(df_who["Date"]))
    df_who["E-I"] = df_who["Pd_Confirmed"].rolling(window=f"{LAG_INFECTED}D", min_periods=1).sum()
    df_who = df_who["E-I"]

    # Replace E-I column in country data
    df.drop(columns=["E-I", "Date"], inplace=True)
    df = pd.concat([df, df_who], axis=1)
    df["E-I"].interpolate(method="index", inplace=True)
    drop_columns = df.columns.to_list()
    drop_columns.remove("Country"); drop_columns.remove("E-I")
    df.dropna(axis=0, subset=drop_columns, inplace=True)
    df.reset_index(drop=False, names="Date", inplace=True)

    # Export data
    df.to_excel(os.path.join(OUT_DIR, f"{country}.xlsx"), index=False)


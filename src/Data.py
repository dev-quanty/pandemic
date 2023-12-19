import numpy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def read_data(country):
    if(country == "sl"):
        header = ("epi_week", "Sr_Confirmed", "Sr_Probable", "Pd_Confirmed", "Pd_Probable")
        path = r"D:\Download\sl_data.csv"
        tmp_data = pd.read_csv(path, sep=",", index_col=False, names=header)
        return tmp_data
    elif(country == "guinea"):
        header = ("epi_week", "Sr_Confirmed", "Sr_Probable", "Pd_Confirmed", "Pd_Probable")
        #nicht lokal path = r"D:\Download\data.csv"
        tmp_data = pd.read_csv(path, sep=",", index_col=False, names=header)
        return tmp_data
    elif(country == "Liberia"):
        header = ("epi_week", "Sr_Confirmed", "Sr_Probable", "Pd_Confirmed", "Pd_Probable")
        #nicht lokal path = r"D:\Download\data.csv"
        tmp_data = pd.read_csv(path, sep=",", index_col=False, names=header)
        return tmp_data


def remove_sparse(df):
    drop_empty = df.dropna(how="all").fillna(0)
    drop_firstrows = drop_empty.drop([0, 1, 2, 3])
    drop_columns = drop_firstrows.drop(columns=["epi_week", "Sr_Probable", "Pd_Probable"])
    reset_index = drop_columns.reset_index(drop=True)
    trimmed_data = reset_index.iloc[:90]
    return trimmed_data


def split_data(df):
    situation_report_data = df.iloc[:, 0]
    patient_data = df.iloc[:, [1]]
    return situation_report_data, patient_data


def change_datatype(df):
    df.iloc[2:, 2:] = df.iloc[2:, 2:].astype(int)
    return df


def toNumpyArray(df):
    dataArray = df.to_numpy(numpy.asarray())
    return dataArray


# def toSIRandSEIRDandSEIRandseirqhfd


def getData():
    data = read_data();
    data = remove_sparse(data)
    data = change_datatype(data)
    situation_report_data, patient_data = split_data(data)
    return patient_data



import os
import pandas as pd


def create_readable_data():
    directory = '...\\ebola-master\\liberia_data'
    df_all = pd.DataFrame()

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, filename),
                             skiprows=[i for i in range(1, 20) if i not in [8, 11, 17, 18]], usecols=[2], nrows=4)
            df_all = pd.concat([df_all, df.T])
    return (df_all)


if __name__ == "__main__":
    create_readable_data().to_csv('liberia_data_reworked.csv', index=False)

    """The result has
        Total death/s in confirmed, probable, suspected cases - 8
        Total contacts listed - 11
        Total no. currently in Treatment Units - 17
        Total discharges - 18
    """

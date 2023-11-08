import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(filename):
    header = ("epi week", "Sr_Confirmed", "Sr_Probable", "Pd_Confirmed", "Pd_Probable")
    path = f"Data/{filename}.csv"
    tmp_data = pd.read_csv(path, sep=",", index_col=False, names=header)
    tmp_data["epi week"] = tmp_data["epi week"].astype(str).apply(lambda x: x.split("(")[-1].rstrip(")"))
    return tmp_data


total_cases = pd.DataFrame([])
countries = ("SierraLeone", "Liberia", "Guinea")
for country in countries:
    filename = country.lower() + "_total"
    country_total = read_data(filename).iloc[4:].reset_index(drop=True)
    country_total = country_total.fillna(0)
    country_total = country_total[["epi week", "Pd_Confirmed"]]
    country_total.columns = ["epi week", country]
    country_total[country] = country_total[country].astype(int)
    if total_cases.empty: total_cases = country_total
    else: total_cases = pd.merge(left=total_cases, right=country_total, how="outer", on="epi week")

fig, ax = plt.subplots()
cases = [total_cases[country] for country in countries]
ax.stackplot(total_cases["epi week"], *cases, labels=countries, colors=sns.color_palette("viridis", 3))
ax.set_xticks(ax.get_xticks()[::10])
plt.xticks(rotation=45, ha='right')

plt.ylabel("# Cases")
plt.legend()
plt.tight_layout()
plt.savefig("Images/TotalCases.png")
plt.show()

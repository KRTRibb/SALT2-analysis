import pandas as pd

tns_sn_df = pd.read_csv("Swinburne/TNS search data/tns_search_combined.csv")
maskZTF = tns_sn_df["Disc. Internal Name"].str.contains("ZTF", na=False)

tns_sn_df = tns_sn_df[maskZTF]
tns_sn_ids = tns_sn_df["Disc. Internal Name"]

flux_fits_df = pd.read_csv(f"/Users/kai/Desktop/Coding/Python/Swinburne/flux_plots_50.0%_confidence/evaluation_updated.csv")


flux_fits_df = flux_fits_df.merge(
    tns_sn_df[["Disc. Internal Name", "Obj. Type", "Redshift"]],
    how="left",
    left_on="object id",
    right_on="Disc. Internal Name"
)

flux_fits_df["TNS classified"] = flux_fits_df["Obj. Type"]

flux_fits_df["redshift"] = flux_fits_df["Redshift"]

flux_fits_df = flux_fits_df.drop(columns=["Disc. Internal Name", "Obj. Type", "Redshift"])

flux_fits_df.to_csv("flux_fits_data.csv")

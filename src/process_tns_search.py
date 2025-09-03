import pandas as pd
import config

def tns_search_process():

    files = [f"tns_search{i}.csv" for i in range(1, config.NUM_TNS_SEARCH_FILES + 1)]

    tns_sn_df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

    tns_sn_df.to_csv(config.TNS_SEARCH_DATA_DIR / "tns_search_combined.csv")
    maskZTF = tns_sn_df["Disc. Internal Name"].str.contains("ZTF", na=False)

    tns_sn_df = tns_sn_df[maskZTF]
    tns_sn_ids = tns_sn_df["Disc. Internal Name"]

    flux_fits_df = pd.read_csv(config.RAW_DATA / "flux_fits_initial.csv")


    flux_fits_df = flux_fits_df.merge(
        tns_sn_df[["Disc. Internal Name", "Obj. Type", "Redshift"]],
        how="left",
        left_on="object id",
        right_on="Disc. Internal Name"
    )

    flux_fits_df["TNS classified"] = flux_fits_df["Obj. Type"]

    flux_fits_df["redshift"] = flux_fits_df["Redshift"]

    flux_fits_df = flux_fits_df.drop(columns=["Disc. Internal Name", "Obj. Type", "Redshift"])

    flux_fits_df.to_csv(config.RAW_DATA / "flux_fits_data.csv")

if __name__ == '__main__':
    tns_search_process()
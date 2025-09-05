import flux_fits.sigmoid_power_fits as spf
import sys, os
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

print("imported")
def plot_flux_fits():   
    start_date = config.START_DATE
    end_date = config.END_DATE

    object_ids = spf.get_object_ids(start_date, end_date)
    print(len(object_ids))

    spf.plot_lc_and_fit_sigmoid_power_and_save(object_ids, start_date, end_date)  

if __name__ == "__main__":
    start = time.time()
    plot_flux_fits()
    print(f"Time taken to run flux_fits {time.time() - start}")
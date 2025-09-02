import sigmoid_power_fits as spf
import config

start_date = config.START_DATE
end_date = config.END_DATE

object_ids = spf.get_object_ids(start_date, end_date)

spf.plot_lc_and_fit_sigmoid_power_and_save(object_ids, start_date, end_date)  
# load libraries
import pandas as pd
from ydata_profiling import ProfileReport

# name report
report_filename = "data_report.html"

# read framingham.csv into a pandas dataframe
df = pd.read_csv("framingham.csv")

# use pandas_profiling.ProfileReport to create a report
profile = ProfileReport(df, title="Framingham Heart Study Data Report")

# save report to data_report.html
profile.to_file(output_file=report_filename)
print(f"Wrote report to {report_filename}")

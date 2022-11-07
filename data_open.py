import pandas as pd
from datetime import date

filename = './data/data.h5'
orig_filename = "./data/data.csv"
#변환중....
"""
store = pd.HDFStore(filename)
df_reader = pd.read_csv(orig_filename, chunksize=65536, encoding='euc-kr', iterator=True, header = 0, names = ["date", "code", "time", "start_id", "start_name", "end_id", "end_name", "total_n", "usage_time", "usage_dist"], usecols=["date", "code", "time", "start_id", "end_id", "total_n", "usage_time", "usage_dist"])

genesis = date(2021, 1, 1).toordinal()

for chunk in df_reader:
    chunk["start_id"] = chunk["start_id"].map(lambda x: int(x[3:]))
    chunk["end_id"] = chunk["end_id"].map(lambda x: int(x[3:]))
    chunk["date"] = chunk["date"].map(lambda x: date(x // 10000, (x % 10000) // 100, x % 100).toordinal() - genesis)
    store.append('data', chunk)

store.close()
"""

s = pd.read_csv(orig_filename, nrows=5, encoding='euc-kr', header = 0, names = ["date", "code", "time", "start_id", "start_name", "end_id", "end_name", "total_n", "usage_time", "usage_dist"], usecols=["date", "code", "time", "start_id", "end_id", "total_n", "usage_time", "usage_dist"])
print(s)

# def extract(row):
#     row.start_id = row.start_id[3:]
#     return row
# s.apply(extract, axis= "column")
# for val in s['start_id']:
#     s = s.replace([val], int(val[3:]))
# for val in s['end_id']:
#     s = s.replace([val], int(val[3:]))
# pd.s.to_csv('./data/revised_data.csv')
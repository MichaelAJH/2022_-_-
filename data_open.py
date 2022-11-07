import pandas as pd

filename = './data/data.h5'
orig_filename = "./data/data.csv"
#변환중....


store = pd.HDFStore(filename)
df_reader = pd.read_csv(orig_filename, chunksize=256, encoding='euc-kr', iterator=True, header = 0, names = ["date", "code", "time", "start_id", "start_name", "end_id", "end_name", "total_n", "usage_time", "usage_dist"], usecols=["date", "code", "time", "start_id", "end_id", "total_n", "usage_time", "usage_dist"])

for chunk in df_reader:
    for val in chunk['start_id']:
        chunk = chunk.replace([val], int(val[3:]))
    for val in chunk['end_id']:
        chunk = chunk.replace([val], int(val[3:]))
    print(chunk)
    store.append('data', chunk)

store.close()
"""

s = pd.read_csv(orig_filename, nrows=5, encoding='euc-kr', header = 0, names = ["date", "code", "time", "start_id", "start_name", "end_id", "end_name", "total_n", "usage_time", "usage_dist"], usecols=["date", "code", "time", "start_id", "end_id", "total_n", "usage_time", "usage_dist"])
print(s["end_id"][0])
"""
# def extract(row):
#     row.start_id = row.start_id[3:]
#     return row
# s.apply(extract, axis= "column")
# for val in s['start_id']:
#     s = s.replace([val], int(val[3:]))
# for val in s['end_id']:
#     s = s.replace([val], int(val[3:]))
# pd.s.to_csv('./data/revised_data.csv')
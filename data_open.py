import pandas as pd
import copy
from datetime import date

test_filename = "./data/test_data_loc.h5"
train_filename = "./data/train_data_loc.h5"
orig_filename = "./data/data.csv"
location_filename = "./data/location_data.csv"

#변환중....
store = pd.HDFStore(test_filename)
df_reader = pd.read_csv(orig_filename, chunksize= 65536 * 8, encoding='euc-kr', iterator=True, header = 0, names = ["date", "code", "time", "start_id", "start_name", "end_id", "end_name", "total_n", "usage_time", "usage_dist"], usecols=["date", "code", "time", "start_id", "end_id", "total_n", "usage_time", "usage_dist"])

location_df = pd.read_csv(location_filename, encoding="euc-kr", header = 0, names = ["id", "addr1", "addr2", "x", "y"], usecols=["id", "x", "y"])
location_df = location_df[location_df.x != 0]
location_df["id"] = location_df["id"].map(lambda x: (int(x[3:])))
location_df["y"] = location_df["y"].map(lambda x: (x - 126.798042) * 1000)
location_df["x"] = location_df["x"].map(lambda x: (x - 37.403549) * 1000)

genesis = date(2021, 1, 1).toordinal()
ind = 0

n_id = 0
id_dict = {}

for chunk in df_reader:
    chunk["start_id"] = chunk["start_id"].map(lambda x: (int(x[3:])))
    chunk["end_id"] = chunk["end_id"].map(lambda x: (int(x[3:])))
    chunk["date"] = chunk["date"].map(lambda x: date(x // 10000, (x % 10000) // 100, x % 100).toordinal() - genesis)
    chunk = pd.merge(chunk, location_df.rename(columns={"x" : "start_x", "y" : "start_y"}), how = "inner", left_on="start_id", right_on="id")
    chunk = pd.merge(chunk, location_df.rename(columns={"x" : "end_x", "y" : "end_y"}), how = "inner", left_on="end_id", right_on="id")
    chunk = chunk.drop(["start_id", "end_id", "id_x", "id_y"], axis = 1)
    #store.append("data", chunk)
    if ind == 0:
        print(chunk)
        break
    ind += 1
    if ind == 10:
        store.close()
        store = pd.HDFStore(train_filename)
    
store.close()
print(ind)
exit(0)

#
s = pd.read_csv(orig_filename, nrows=3000, encoding='euc-kr', header = 0, names = ["date", "code", "time", "start_id", "start_name", "end_id", "end_name", "total_n", "usage_time", "usage_dist"], usecols=["date", "code", "time", "start_id", "end_id", "total_n", "usage_time", "usage_dist"])
print(s[0:5])
# def extract(row):
#     row.start_id = row.start_id[3:]
#     return row
# s.apply(extract, axis= "column")
# for val in s['start_id']:
#     s = s.replace([val], int(val[3:]))
# for val in s['end_id']:
#     s = s.replace([val], int(val[3:]))
# pd.s.to_csv('./data/revised_data.csv')
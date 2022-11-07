import pandas as pd
from datetime import date

test_filename = "./data/test_data.h5"
train_filename = "./data/train_data.h5"
orig_filename = "./data/data.csv"
#변환중....

store = pd.HDFStore(test_filename)
df_reader = pd.read_csv(orig_filename, chunksize=65536 * 8, encoding='euc-kr', iterator=True, header = 0, names = ["date", "code", "time", "start_id", "start_name", "end_id", "end_name", "total_n", "usage_time", "usage_dist"], usecols=["date", "code", "time", "start_id", "end_id", "total_n", "usage_time", "usage_dist"])

genesis = date(2021, 1, 1).toordinal()
ind = 0

n_id = 0
id_dict = {}

def encode_id(id):
    global n_id, id_dict
    if id in id_dict:
        return id_dict[id]
    else:
        n_id += 1
        id_dict[id] = n_id
        return n_id

for chunk in df_reader:
    chunk["start_id"] = chunk["start_id"].map(lambda x: encode_id(int(x[3:])))
    chunk["end_id"] = chunk["end_id"].map(lambda x: encode_id(int(x[3:])))
    chunk["date"] = chunk["date"].map(lambda x: date(x // 10000, (x % 10000) // 100, x % 100).toordinal() - genesis)
    store.append("data", chunk)
    ind += 1

    if ind == 10:
        store.close()
        store = pd.HDFStore(train_filename)
    
store.close()
print(ind)
exit(0)

s = pd.read_csv(orig_filename, nrows=10000, encoding='euc-kr', header = 0, names = ["date", "code", "time", "start_id", "start_name", "end_id", "end_name", "total_n", "usage_time", "usage_dist"], usecols=["date", "code", "time", "start_id", "end_id", "total_n", "usage_time", "usage_dist"])
print(s["time"].unique())

# def extract(row):
#     row.start_id = row.start_id[3:]
#     return row
# s.apply(extract, axis= "column")
# for val in s['start_id']:
#     s = s.replace([val], int(val[3:]))
# for val in s['end_id']:
#     s = s.replace([val], int(val[3:]))
# pd.s.to_csv('./data/revised_data.csv')
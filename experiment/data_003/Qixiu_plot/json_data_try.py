import json

with open('/Users/fqx/Dropbox/Spring 2020/Ma Lab/GitHub/Road_Construction/experiment/data_003/test_all_3') as f:
  data = json.load(f)

print("mapid:", type(data))
# print(json.dumps(data, sort_keys=True, indent=20))
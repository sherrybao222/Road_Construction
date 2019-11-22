import pandas as pd

data = pd.read_csv("trial.csv") # access the 4th_col of city_xy from file = pd.read_csv("trial.csv", usecols=[3])
# data = data.iloc[0]
trial = 10
#
for trial in range(0, trial):
    # trial_i = trial.__index__()
    if input("next trial") == str(1):
        data_row = data.loc[trial]
        data_col = data_row[['City_xy']]
        print(data_col)

    # if input("next trial") == 1:
    #     # data = pd.read_csv("trial.csv")
    #     data = data.loc[trial:]
    #     # data = data.loc[trial_i]
    #     print(data)
    # else:
    #     break

# end = True
#
# for trial in data:
#     trial = data[0: ]
#     if end is True:
#         trial_next = data[]
#         print(trial)

# print(data.loc[0])




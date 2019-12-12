import csv
import numpy as np
# =============================================================================
obj = trials[0]
members = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")]
values = [getattr(trials, member) for member in members]
# 
# =============================================================================
# =============================================================================

attrs = [o.time for o in trials]

flat_list = [item for sublist in attrs for item in sublist]

# =============================================================================


data1 = np.arange(10)
data2 = np.arange(10)*2
data3 = np.arange(10)*3

writefile = 'test.csv'
fieldnames = ['data1','data2', 'data3']
with open( writefile, 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    writer.writerows(zip(data1, data2, data3))
    
for member in members:
    attrs = [o.member for o in trials]
    
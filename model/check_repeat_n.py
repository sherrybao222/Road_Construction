import json
import numpy as np

# directories
home_dir = '/Users/dbao/google_drive/'
input_dir = 'road_construction/experiments/pilot_0320/data/data_pilot_preprocessed/'
output_dir = 'road_construction/experiments/pilot_0320/data/data_pilot_preprocessed/ll/'
map_dir = 'road_construction/experiments/pilot_0320/map/active_map/'  

sub = 2

with open(home_dir + output_dir + 'n_repeat_6000_' + str(sub),'r') as file:
    repeats_6 = json.load(file) 

with open(home_dir + output_dir + 'n_repeat_2000_' + str(sub),'r') as file:
    repeats_2 = json.load(file) 

diff = [x - y for x,y in zip(repeats_6[0],repeats_2)]

data = [-387, -396, -397]
#[-392, -398, -391];# [-401.0, -385.8, -373.8]
# First quartile (Q1) 
Q1 = np.percentile(data, 25, interpolation = 'midpoint') 
  
# Third quartile (Q3) 
Q3 = np.percentile(data, 75, interpolation = 'midpoint') 
  
# Interquaritle range (IQR) 
IQR = Q3 - Q1 
print(IQR)

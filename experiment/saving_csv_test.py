import csv
import numpy as np
# =============================================================================

members = [attr for attr in dir(trials[0]) if not callable(getattr(trials[0], attr)) and not attr.startswith("__")]
attrs = []
for member in members:
    attr = [getattr(trial, member) for trial in trials]
    if member != city_start and member != distance:
        try: flat_list = [item for sublist in attr for item in sublist]
        except: flat_list = attr
    else: 
        flat_list = attr
    attrs.append(flat_list)
# 
# =============================================================================
# =============================================================================

attrs = [o.time for o in trials]

flat_list = [item for sublist in attrs for item in sublist]

# =============================================================================


data1 = np.arange(10)
data2 = np.arange(10)*2
data3 = np.arange(10)*3

blk = trials.blk
trials.cond = [3] # condition
trials.time = [round((pg.time.get_ticks()/1000), 2)] # mouse click time 
trials.pos = [pg.mouse.get_pos()]
trials.click = [0] # mouse click indicator
self.undo_press = [0] # undo indicator

self.choice_dyn = [0]
self.choice_locdyn = [self.city_start]
self.choice_his = [0]   # choice history, index
self.choice_loc = [self.city_start] # choice location history
        
self.budget_dyn = [self.total]
self.budget_his = [self.total] # budget history

self.n_city = 0 # number of cities connected
self.check = 0 # indicator showing if people made a valid choice
self.num_est = [None] # number estimation input()

writefile = 'test.csv'
fieldnames = ['data1','data2', 'data3']
with open( writefile, 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    writer.writerows(zip(data1, data2, data3))
    
for member in members:
    attrs = [getattr(o, member) for o in trials]
    
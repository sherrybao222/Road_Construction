import csv
# =============================================================================

members = [attr for attr in dir(trials[0]) if not callable(getattr(trials[0], attr)) and not attr.startswith("__")]
exp = ['blk','cond','trl','time','pos','click','undo_press','choice_his','choice_loc',
              'budget_his','n_city','num_est']
info = ['N','R','phi','r','radius','total','x','y','xy','city_start','distance','order']

#exps = []
#infos = []
dict_exp = {}
dict_info = {}

for member in members:
    attr = [getattr(trial, member) for trial in trials]
    if member in set(exp):
        flat_list = [item for sublist in attr for item in sublist]
#        exps.append(flat_list)
        dict_exp[member] = flat_list        
    else:
        dict_info[member] = attr
#        infos.append(attr)
        
writefile = 'test_exp.csv'
with open( writefile, 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow(exp)
    all_values = [dict_exp[exp[i]] for i in range(len(exp))]
    writer.writerows(zip(*all_values))

writefile = 'test_info.csv'
with open( writefile, 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow(info)
    all_values = [dict_info[info[i]] for i in range(len(info))]
    writer.writerows(zip(*all_values))
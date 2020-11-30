from llh_ibs_general_repeats import ibs_grepeats
from bads_prepare import sub_data,repeats,basic_map,LL_lower # import x and y data
import json

# hpc directories
home_dir = '/home/db4058/road_construction/data/'

def ibs_interface(w1, w2, w3, w4, w5, w6, w7):
     '''
     interface used to connect BADS and ibs_repeat
     '''
     
     inparams = [w1, w2, w3, w4, w5, w6, w7]
     print('inparams '+str(inparams))
     nll_avg, time_sequence,count_iteration =ibs_grepeats(inparams,LL_lower,sub_data,basic_map,repeats)
 	
     return nll_avg,time_sequence,count_iteration



if __name__ == "__main__":   
    nll, time_seq, n_sample = ibs_interface(1, 1, 1, 0.01, 15, 0.05, 0.1)
    
    with open(home_dir + 'check_time_sample','w') as file: 
        json.dump({'time sequence':time_seq,
               'number of samples':n_sample}, file)
    
    


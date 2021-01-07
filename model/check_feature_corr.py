import pandas as pd
from scipy.stats import shapiro
import numpy as np

## directories
home_dir = '/Users/dbao/google_drive/'
input_dir = 'road_construction/experiments/pilot_0320/data/data_pilot_preprocessed/'
map_dir = 'road_construction/experiments/pilot_0320/map/active_map/'

subs = [1,2,4] # subject index 
corr_bu = [np.nan]*3
corr_cu = [np.nan]*3
corr_bc = [np.nan]*3

for idx, sub in enumerate(subs):
    subject_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_'+str(sub) + '.csv')
    
    budget_remain = subject_data.loc[:,'budget_all']
    n_city = subject_data.loc[:,'n_city_all']
    n_u = subject_data.loc[:,'n_u_all']

    # # normality test
    # stat, p = shapiro(n_u)
    # print('Statistics=%.3f, p=%.3f' % (stat, p))
    # # interpret
    # alpha = 0.05
    # if p > alpha:
    #     	print('Sample looks Gaussian (fail to reject H0)')
    # else:
    #     	print('Sample does not look Gaussian (reject H0)')
            
    corr_bu[idx] = budget_remain.corr(n_u, method='spearman')  # Spearman's rho
    corr_cu[idx] = n_city.corr(n_u, method='spearman')  
    corr_bc[idx] = budget_remain.corr(n_city, method='spearman')  
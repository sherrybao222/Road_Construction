# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from scipy.stats import shapiro
from scipy.stats import normaltest

home_dir = '/Users/dbao/google_drive_db'+'/road_construction/data/2022_online/'
map_dir = 'active_map/'
data_dir  = 'data/preprocessed'
out_dir = 'figures/figures_all/'
R_out_dir = home_dir + 'R_analysis_data/'

data_puzzle_level = pd.read_csv(R_out_dir +  'data.csv')
puzzleID_order_data = data_puzzle_level.sort_values(["subjects","puzzleID"])
data_choice_level = pd.read_csv(R_out_dir +  'choice_level/choicelevel_data.csv')


# +
def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))

# add p-value to figure
def text(p):
    if p < 0.001:
        axs[i].text((x1+x2)*.5, y+h, r"$p = {0:s}$".format(as_si(p,1)), ha='center', va='bottom', color=col, fontsize = 16)
    elif p > 0.1:
        axs[i].text((x1+x2)*.5, y+h, r"$p = {:.2f}$".format(p), ha='center', va='bottom', color=col, fontsize = 16)

    elif 0.01 < p < 0.1:
        axs[i].text((x1+x2)*.5, y+h, r"$p = {:.3f}$".format(p), ha='center', va='bottom', color=col, fontsize = 16)
    else:
        axs[i].text((x1+x2)*.5, y+h, r"$p = {:.4f}$".format(p), ha='center', va='bottom', color=col, fontsize = 16)


# -

index_start = data_choice_level.index[data_choice_level['RT'] == -1]
RT_first_move = data_choice_level.loc[index_start+1,:]
index_later = data_choice_level.index[(data_choice_level['RT'] != -1) & (data_choice_level['submit'] != 1)& (data_choice_level['undo'] != 1)]
RT_later_move = data_choice_level.loc[index_later,:]
index_submit = data_choice_level.index[data_choice_level['submit'] == 1]
RT_submit = data_choice_level.loc[index_submit,:]


# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

# df_part = df.loc[df['sub'] == subs[i],['f_t_rc','f_t_undo',
#                  'm_t_everyact_rc','m_t_everyc_undo',
#                  't_s_rc','t_s_undo']]

# undobox = []
# for x in t_everyundo[48*i:48*(i+1)]:
#     try: 
#         undobox.append(median(x))
#     except:  pass

# axs[i].plot([1,2],[f_t_rc[48*i:48*(i+1)], f_t_undo[48*i:48*(i+1)]],c = '#6a6763',linewidth=0.3) 
# axs[i].plot([2.5,3.5],[[median(x) for x in t_everyact_rc[48*i:48*(i+1)]], [median(x) for x in t_everyc_undo[48*i:48*(i+1)]]],c = '#6a6763',linewidth=0.3) 
# axs[i].plot([4,5],[t_s_rc[48*i:48*(i+1)],t_s_undo[48*i:48*(i+1)]],c = '#6a6763',linewidth=0.3) 


bx = axs.boxplot([RT_first_move[RT_first_move['condition']==0]['RT'],RT_first_move[RT_first_move['condition']==1]['RT'],
    RT_later_move[RT_later_move['condition']==0]['RT'],RT_later_move[RT_later_move['condition']==1]['RT'],
    RT_submit[RT_submit['condition']==0]['RT'],RT_submit[RT_submit['condition']==1]['RT']],
#    [median(x) for x in t_everyc_undo[48*i:48*(i+1)]],
#    t_s_rc[48*i:48*(i+1)],t_s_undo[48*i:48*(i+1)],
#    undobox],
   positions =[1,2,3.5,4.5,6,7],widths = 0.3,showfliers=False,whis = 1.5,
   medianprops = dict(color = 'k'))  #
    
# normality test
# Shapiro-Wilk Test
    
# stats = [np.nan]*6
# ps = [np.nan]*6
# for s in range(len(stats)):
#     stats[s],ps[s]  = shapiro([math.log2(x) for x in df_part.iloc[:,s]])
#     print('Statistics=%.3f, p=%.3f' % (stats[s], ps[s]))
#     # interpret
#     alpha = 0.05
#     if ps[s] > alpha:
#         print('Sample looks Gaussian (fail to reject H0)')
#     else:
#         print('Sample does not look Gaussian (reject H0)')   
    
#     from scipy.stats import anderson
#     result = [np.nan]*6
#     for s in range(len(result)):
#         result[s]= anderson([math.log2(x) for x in df_part.iloc[:,s]])
#         print('Statistic: %.3f' % result[s].statistic)
#         p = 0
#         for i in range(len(result[s].critical_values)):
#         	sl, cv = result[s].significance_level[i], result[s].critical_values[i]
#         	if result[s].statistic < result[s].critical_values[i]:
#         		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
#         	else:
#         		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
    
#     # statistical annotation
#     stat1, p1 = wilcoxon(df_part['f_t_rc'], df_part['f_t_undo'])
#     x1, x2 = 1,2  
#     if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
#         y, h, col = bx['caps'][1]._y[0] + 2, 2, 'k'
#     else:
#         y, h, col = bx['caps'][3]._y[0] + 2, 2, 'k'
    
#     axs[i].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
#     text(p1)
    
#     #--------------------------------------
#     stat2, p2 = wilcoxon(df_part['m_t_everyact_rc'], df_part['m_t_everyc_undo'])

#     x1, x2 = 2.5,3.5  
#     y, h, col = bx['caps'][5]._y[0] + 2, 2, 'k'
#     axs[i].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
#     text(p2)
#     #--------------------------------------
#     stat3, p3 = wilcoxon(df_part['t_s_rc'], df_part['t_s_undo'])

#     x1, x2 = 4,5  
#     y, h, col = bx['caps'][11]._y[0] + 2, 2, 'k'
#     axs[i].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
#     text(p3)
#     #--------------------------------------
#     stat4, p4 = wilcoxon(df_part['f_t_rc'], df_part['m_t_everyact_rc'])

#     x1, x2 = 1,2.5  
#     if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
#         y, h, col = bx['caps'][1]._y[0] + 6, 2, 'k'
#     else:
#         y, h, col = bx['caps'][3]._y[0] + 6, 2, 'k'
        
#     axs[i].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
#     text(p4)
#     #--------------------------------------
#     stat5, p5 = wilcoxon(df_part['f_t_undo'], df_part['m_t_everyc_undo'])

#     x1, x2 = 2,3.5  
#     if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
#         y, h, col = bx['caps'][1]._y[0] + 12, 2, 'k'
#     else:
#         y, h, col = bx['caps'][3]._y[0] + 12, 2, 'k'
        
#     axs[i].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
#     text(p5)


#     #--------------------------------------

axs.set_xticks([1,1.5,2, 3.5,4,4.5, 6,6.5,7])
axs.set_xticklabels(labels = ['\nwithout \nundo','first choice','\nwith \nundo','\nwithout \nundo','later choices','\nwith \nundo', '\nwithout \nundo','submit','\nwith \nundo'])#,fontsize=18

axs.set_facecolor('white')
axs.spines['bottom'].set_color('k')
axs.spines['left'].set_color('k')
axs.tick_params(axis='y', colors='k', direction='in',left = True) #, labelsize = 16
axs.tick_params(axis='x', colors='k')
# axs.set_title('S'+str(i+1), fontsize = 16)
axs.set_ylabel('Response time (ms)') #,fontsize=18

# fig.set_figwidth(26)
# fig.set_figheight(12)

plt.show()
# fig.savefig(out_dir + 'action_t.png',dpi=600,bbox_inches='tight')
# plt.close(fig)

# +
index_singleUndo = data_choice_level.index[(data_choice_level['firstUndo'] == 1)&(data_choice_level['lastUndo'] == 1)]
RT_singleUndo = data_choice_level.loc[index_singleUndo,:]

index_firstUndo = data_choice_level.index[(data_choice_level['firstUndo'] == 1) &(data_choice_level['lastUndo'] != 1)]
RT_firstUndo = data_choice_level.loc[index_firstUndo,:]
index_laterUndo = data_choice_level.index[(data_choice_level['firstUndo'] != 1) & (data_choice_level['undo'] == 1)]
RT_laterUndo = data_choice_level.loc[index_laterUndo,:]

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

bx = axs.boxplot([RT_firstUndo['undoRT'],RT_laterUndo['undoRT'],RT_singleUndo['undoRT']],
   positions =[1,2,3],widths = 0.3,showfliers=False,whis = 1.5,
   medianprops = dict(color = 'k'))  #

axs.set_xticks([1,1.5,2,3])
axs.set_xticklabels(labels = ['\nfirst undo','sequential','\nlater undo','single undo'])#,fontsize=18

axs.set_facecolor('white')
axs.spines['bottom'].set_color('k')
axs.spines['left'].set_color('k')
axs.tick_params(axis='y', colors='k', direction='in',left = True) #, labelsize = 16
axs.tick_params(axis='x', colors='k')
# axs.set_title('S'+str(i+1), fontsize = 16)
axs.set_ylabel('Response time (ms)') #,fontsize=18

# fig.set_figwidth(26)
# fig.set_figheight(12)

plt.show()
# fig.savefig(out_dir + 'action_t.png',dpi=600,bbox_inches='tight')
# plt.close(fig)

# +
index_first_undo =  data_choice_level.index[data_choice_level['firstUndo'] == 1]
df_beforeUndo = data_choice_level.loc[index_first_undo-1,:]
index_end_undo = df_beforeUndo.index[df_beforeUndo['checkEnd'] == 1]
leftover_undo = df_beforeUndo.loc[index_end_undo,'leftover']


index_notundo = data_choice_level.index[(data_choice_level['undo'] == 0)&(data_choice_level['RT'] != -1)]
df_notbeforeUndo = data_choice_level.loc[index_notundo-1,:]
index_end_notundo = df_notbeforeUndo.index[df_notbeforeUndo['checkEnd'] == 1]
leftover_notundo = df_notbeforeUndo.loc[index_end_notundo,'leftover']


# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

bx = axs.boxplot([leftover_undo,leftover_notundo],
   positions =[1,2],widths = 0.3,showfliers=False,whis = 1.5,
   medianprops = dict(color = 'k'))  #

axs.set_xticks([1,2])
axs.set_xticklabels(labels = ['budget before undo','budget before submit'])#,fontsize=18

axs.set_facecolor('white')
axs.spines['bottom'].set_color('k')
axs.spines['left'].set_color('k')
axs.tick_params(axis='y', colors='k', direction='in',left = True) #, labelsize = 16
axs.tick_params(axis='x', colors='k')
# axs.set_title('S'+str(i+1), fontsize = 16)
axs.set_ylabel('budget') #,fontsize=18

# fig.set_figwidth(26)
# fig.set_figheight(12)

plt.show()
# fig.savefig(out_dir + 'action_t.png',dpi=600,bbox_inches='tight')
# plt.close(fig)
# -



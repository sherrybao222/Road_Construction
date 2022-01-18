from util_figure import plot_figure
from glob import glob
import numpy as np
from numpy import genfromtxt
import pandas as pd
import json
import os

###############################################################################################
home_dir = './'
map_dir = 'active_map/'
data_dir  = 'data/'
out_dir = 'figures_undo/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

fig_class = plot_figure.figure_plott(home_dir=home_dir, map_dir=map_dir, data_dir=data_dir, out_dir = 'figures_basic/')
fig_class.filter_subset(tag_name='basic')
fig_class.rc_severity_of_errors_hist()
fig_class = plot_figure.figure_plott(home_dir=home_dir, map_dir=map_dir, data_dir=data_dir, out_dir = out_dir)
fig_class.filter_subset(tag_name='undo')
# fig_class.rc_undo_nos_puzzle()
fig_class.rc_severity_of_errors_hist()

fig_class.rc_undo_severity_of_puzzle_errors_choice()
fig_class.rc_undo_severity_of_errors_choice()
fig_class.rc_undo_severity_of_errors_puzzle()
# fig_class.rc_undo_leftover_puzzle()
fig_class.rc_undo_nct_choice()
fig_class.rc_undo_nct_puzzle()
fig_class.rc_undo_nos_choice()
fig_class.rc_undo_mas_choice()


fig_class.rc_nct_undo_puzzle(5)
fig_class.rc_severity_of_errors_undo_puzzle(5)
fig_class.rc_undo_leftover_choice()
fig_class.rc_undo_mas_puzzle()
fig_class.rc_undo_nos_hist_puzzle(5)
fig_class.rc_undo_nct_progress_choice()
fig_class.rc_undo_mas_errors_choice()
fig_class.rc_undo_nos_hist_choice()
# fig_class.rc_undo_severity_of_errors_choice()


###############################################################################################
home_dir = './'
map_dir = 'active_map/'
data_dir  = 'data/'
out_dir = 'figures_all/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

fig_class = plot_figure.figure_plott(home_dir=home_dir, map_dir=map_dir, data_dir=data_dir, out_dir = out_dir)
fig_class.rc_undo_hist_all_across_subjects_ag()
fig_class.rc_undo_hist_all_across_subjects()
fig_class.rc_undo_c_numciti()




fig_class.rc_severe_error_undo()
fig_class.rc_error_undo_length()
fig_class.rc_undo_x_mas()





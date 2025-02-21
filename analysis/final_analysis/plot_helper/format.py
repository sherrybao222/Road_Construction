import matplotlib.pyplot as plt
# Set the style to remove top and right borders
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
# set the font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 25})
# remove legend edge
plt.rcParams['legend.frameon'] = False
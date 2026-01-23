# add p-value to figure
def text(p, axs, x1, x2, y, h, col):
    if (0.001 < p < 0.01)|(p == 0.001):
        axs.text((x1+x2)*.5, y+h,  r"**", ha='center', va='bottom', color=col, fontsize = 8)
    elif p < 0.001:
        axs.text((x1+x2)*.5, y+h, r"***", ha='center', va='bottom', color=col, fontsize = 8)
    elif (0.01 < p < 0.05)|(p == 0.01):
        axs.text((x1+x2)*.5, y+h, r"*", ha='center', va='bottom', color=col, fontsize = 8)
    else:
        axs.text((x1+x2)*.5, y+h, r"n.s.", ha='center', va='bottom', color=col, fontsize = 8)
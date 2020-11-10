from ibs_general_repeats import ibs_grepeats
from prepare import sub_data,repeats,basic_map,LL_lower # import x and y data

def ibs_interface(w1, w2, w3, w4, w5, w6, w7):
 	'''
 	interface used to connect BADS and ibs_repeat
 	'''
 	inparams = [w1, w2, w3, w4, w5, w6, w7]
 	print('inparams '+str(inparams))
    
 	return ibs_grepeats(inparams,LL_lower,sub_data,basic_map,repeats)



if __name__ == "__main__":   
    ibs_interface(1, 1, 1, 0.01, 15, 0.05, 0.1)


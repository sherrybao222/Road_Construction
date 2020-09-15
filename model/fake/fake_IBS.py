
'''
fake log likelihood script for testing fitting code
'''
import numpy as np
import scipy.stats as stats


class Params:
	def __init__(self, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11):
		self.weights = [w1, w2, w3, w4, w5, w6, w7, w8, w9]
		self.intercept = w10
		self.sd = w11



def ibs_repeat(inparams,  
			subject_data,
			subject_answer): 
	'''
		ibs without early stopping
		sequential
		with trial-dependent repeated sampling
		returns the log likelihood of current subject dataset
	'''
	params = Params(w1=inparams[0], w2=inparams[1], w3=inparams[2], w4=inparams[3], w5=inparams[4],
					w6=inparams[5], w7=inparams[6], w8=inparams[7], w9=inparams[8], w10=inparams[9], w11=inparams[10])
	yhat = params.intercept + np.matmul(subject_data, params.weights) # prediction
	LL = np.sum(stats.norm.logpdf(subject_answer, loc=yhat, scale=params.sd)) # get log likelihood for all trials
	print('inparams: '+str(inparams)+', LL='+str(LL))
	return -LL # return negative ll



from fake_prepare import subject_data, subject_answer # import x and y data


def ibs_interface(w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11):
	'''
	interface used to connect BADS and ibs_repeat
	'''
	inparams = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11]
	print('inparams '+str(inparams))
	return ibs_repeat(inparams,  
					subject_data,
					subject_answer)










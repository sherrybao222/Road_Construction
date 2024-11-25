'''
fake prepare script, for testing BADS interface
'''
import numpy as np

def prepare_ibs(subject_file='',
				subject_path='/Users/yichen/Desktop/subjects/',
				preprocessed_data_path='/Users/yichen/Desktop/preprocessed_positions/',
				instancepath='/Users/yichen/Documents/RushHour/exp_data/data_adopted/'): # sequantial
	'''
	generate data 
	parameters to be fitted: feature weights, noise sd, intercept
	'''
	N = 2400 # sample size
	x = np.random.rand(N, 9) # x data with 9 features
	noise = np.random.normal(loc=0.0, scale=5.0, size=N) # noise term, sd to be fitted
	y = np.matmul(x, np.array([9,8,7,6,5,4,3,2,1]))+noise # generate y with correct feature weights
	# total 9+1+1 paramaters to be fitted

	return x, y


# subject_file = random.choice(os.listdir(home_dir+'Desktop/subjects/'))

# subject_file = 'A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9.csv'
# subject_file = 'A1EY7WONSYGBVY:3O6CYIULEE9B1LG2ZMEYM39PDJKUWB.csv'
subject_file = 'A1F4N58CAX8IMK:35DR22AR5ES6RR89U7EJ1DXW91AX3P.csv'
# subject_file = 'A1GQS6USF2JEYG:33F859I567LE8WC74WB3GA7E94XBHW.csv'
# subject_file = 'A1LR0VQIHQUJAM:3YW4XOSQKRTI0K0Z2YPDTDJVFABU1I.csv'
# subject_file = 'A2TQNX64349OZ9:3WSELTNVR4AZUVYAYCSWZIQW0J4ATB.csv'

home_dir = '/Users/yichen/'
instancepath=home_dir+'Documents/RushHour/exp_data/data_adopted/'
preprocessed_data_path = home_dir+'Desktop/preprocessed_positions/'
subject_data, subject_answer = prepare_ibs(subject_file=subject_file)




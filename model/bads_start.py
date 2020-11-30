import matlab.engine
'''
start to fit with BADS!
this script call BADS from python matlab engine,
BADS will call a matlab fake_ll.m script, which calls python fake_IBS.py
'''

if __name__ == '__main__':
	x0 = matlab.double([1, 1, 1, 0.1, 10, 0.1, 0.1]) # starting point
	lb = matlab.double([0, 0, 0, 0, 0.1, 0, 0]) # strict lower bound
	ub = matlab.double([10, 10, 10, 1, 30, 1, 1]) # strict upper bound
	plb = matlab.double([0, 0, 0, 0, 0.1, 0.001, 0]) # plausible lower bound
	pub = matlab.double([10, 10, 10, 1, 20, 0.5, 0.5]) # plausible upper bound

	eng = matlab.engine.start_matlab()

	[x,fval] = eng.bads("@bads_ll", x0,lb,ub,plb,pub) 
	print('result', [x,fval])


import matlab.engine
'''
fake starting script to test BADS interface
start to fit with BADS!
this script call BADS from python matlab engine,
BADS will call a matlab fake_ll.m script, which calls python fake_IBS.py
'''

if __name__ == '__main__':
	x0 = matlab.double([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 2.0, 1.0]) # starting point
	lb = matlab.double([-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,1]) # strict lower bound
	ub = matlab.double([20,20,20,20,20,20,20,20,20,20,20]) # strict upper bound
	plb = matlab.double([0,0,0,0,0,0,0,0,0,0,1]) # plausible lower bound
	pub = matlab.double([15,15,15,15,15,15,15,15,15,15,15]) # plausible upper bound

	eng = matlab.engine.start_matlab()
	# sys.setrecursionlimit(31000)
	options = matlab.double([])
	# options.TolMesh = '1e-8'
	# options.TolFun = '1e-5'
	result = eng.bads("@fake_ll", x0,lb,ub,plb,pub, options, nargout=2) # BADS will call fake_ll.m to get neg LL
	print('result[0]', result[0])
	print('result[1]', result[1])

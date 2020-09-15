x0 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 2.0, 1.0];
lb = [-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,1];
ub = [20,20,20,20,20,20,20,20,20,20,20];
plb = [0,0,0,0,0,0,0,0,0,0,1];
pub = [15,15,15,15,15,15,15,15,15,15,15];

[x,fval] = bads(@fake_ll,x0,lb,ub,plb,pub);

function y = fake_ll(x)
%ROSENBROCKS Rosenbrock's 'banana' function in any dimension.

y = py.fake_IBS.ibs_interface(x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), x(9), x(10), x(11));
disp(y);

end
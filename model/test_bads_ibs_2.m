x0 = [1, 1, 1, 0.01, 15, 0.05, 0.1];    % Starting point
lb = [0.01, 0.01, 0.01, 0.00001, 2, 0.00001, 0.00001];   % Lower bounds
ub = [10, 10, 10, 0.5, 30, 0.5, 0.5];   % Upper bounds
plb = [0.1, 0.1, 0.1, 0.001, 5, 0.01, 0.01];   % Plausible lower bounds
pub = [5, 5, 5, 0.3, 20, 0.3, 0.3];   % Plausible upper bounds

[x,fval] = bads(@bads_ll,x0,lb,ub,plb,pub);

function y = bads_ll(x)

    y = py.start.ibs_interface(x(1), x(2), x(3), x(4), x(5), x(6), x(7));
    disp(y);
end
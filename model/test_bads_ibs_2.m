diary myDiaryFile % save output in command window
tic % record time

% w1, w2, w3,stopping_probability,pruning_threshold,lapse_rate,feature_dropping_rate,

x0 = [1, 1, 1, 0.1, 15, 0.1, 0.1];    % Starting point

lb = [0, 0, 0, 0, 0.1, 0, 0];   % Lower bounds
ub = [10, 10, 10, 1, 30, 1, 1];   % Upper bounds

plb = [0, 0, 0, 0, 0.1, 0.001, 0];   % Lower bounds
pub = [10, 10, 10, 1, 30, 0.3, 0.5];   % Upper bounds


[x,fval] = bads(@bads_ll,x0,lb,ub,plb,pub);

toc
diary off

function y = bads_ll(x)

y = py.bads_ibs.ibs_interface(x(1), x(2), x(3), x(4), x(5), x(6), x(7));
disp(y);

end


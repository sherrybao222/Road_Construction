%pyenv('Version','~/.conda/envs/road/bin/python')
addpath('bads-master')

%diary myDiaryFile_2 % save output in command window
tic % record time

% set random seeds
rng ('default')
seed = rng; 

%% w1, w2, w3,stopping_probability,pruning_threshold,lapse_rate,feature_dropping_rate

LB = [0, 0, 0, 0, 0.1, 0, 0];   % Lower bounds
UB = [10, 10, 10, 1, 30, 1, 1];   % Upper bounds

PLB = [0, 0, 0, 0, 0.1, 0.001, 0];   % Lower bounds
PUB = [10, 10, 10, 1, 20, 0.5, 0.5];   % Upper bounds

N_STARTS = 20; % number of different x0

for r = 1:N_STARTS
    nvars = numel(PLB);
    X0 = PLB + rand(1,nvars) .* (PUB - PLB);   % Starting point
    
    [x,fval] = bads(@bads_ll,X0,LB,UB,PLB,PUB);
    
    Output(r).x0 = x0;
    Output(r).x = x;
    Output(r).fval = fval;
end

toc
%diary off




function run_bads()

    myHOME = '/home/db4058/';
    addpath([myHOME, '/toolbox/bads-master'])

    % save output in command window
    %diary myDiaryFile_2 
    tStart = tic; % record time

    % w1, w2, w3,stopping_probability,pruning_threshold,lapse_rate,feature_dropping_rate

    LB = [0, 0, 0, 0, 0.1, 0, 0];   % Lower bounds
    UB = [10, 10, 10, 1, 30, 1, 1];   % Upper bounds

    PLB = [0, 0, 0, 0, 0.1, 0.001, 0];   % Lower bounds
    PUB = [10, 10, 10, 1, 20, 0.5, 0.5];   % Upper bounds

%     nvars = numel(PLB);
%     rand_coe = rand(1,nvars);
%     X0 = PLB + rand_coe .* (PUB - PLB);   % Starting point
    X0 = [1, 1, 1, 0.1, 10, 0.1, 0.1];  

    [x,fval] = bads(@bads_ll,X0,LB,UB,PLB,PUB);

    Output.x0 = x0;
    Output.x = x;
    Output.fval = fval;


    tEnd = toc(tStart);
    %diary off

    save('test.mat')
end




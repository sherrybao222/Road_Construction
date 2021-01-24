function bads_run_greedy(hpc,job_id,task_id,sub)
    % set euler
    % euler = parcluster('local');
    % pool = parpool(euler,1);
    if hpc == 1
    % set directory
        myHOME = '/home/db4058/';
        addpath([myHOME, '/toolbox/bads-master'])
    end

    % save output in command window
    %diary myDiaryFile_2 
    tStart = tic; % record time

    % w1, w2, w3,stopping_probability,pruning_threshold,lapse_rate,feature_dropping_rate

    LB  = [0,  0,  0,  0.1, 0.01, 0];   % Lower bounds
    UB  = [10, 10, 10, 30,  1,    1];   % Upper bounds

    PLB = [0,  0,  0,  0.1, 0.01, 0];   % Lower bounds
    PUB = [10, 10, 10, 20,  0.5,  0.2];   % Upper bounds

    nvars    = numel(PLB);
    rand_coe = rand(1,nvars);
    X0       = PLB + rand_coe .* (PUB - PLB)  % Starting point
    
    funwdata = @(x) bads_ll_greedy(x,sub);
    
    [x,fval] = bads(funwdata,X0,LB,UB,PLB,PUB);

    Output.x0   = X0;
    Output.x    = x;
    Output.fval = fval;


    tEnd = toc(tStart);
    % diary off
    filename = sprintf('/home/db4058/road_construction/result/bads_%.0f_%.0f.mat',job_id,task_id);
    save(filename)

    % pool.delete()
end


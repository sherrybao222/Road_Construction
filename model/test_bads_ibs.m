% --------------------------------------------------------------------
% try run model fitting in matlab
% --------------------------------------------------------------------
% w1, w2, w3, stopping_probability, pruning_threshold, 
% lapse_rate, feature_dropping_rate

x0 = [1, 1, 1, 0.01, 15, 0.05, 0.1];    % Starting point
lb = [0, 0, 0, 0.00001, 2, 0.00001, 0.00001];   % Lower bounds
ub = [10, 10, 10, 0.5, 30, 0.5, 0.5];   % Upper bounds
plb = lb;   % Plausible lower bounds
pub = ub;   % Plausible upper bounds
nonbcon = [];
options = [];


% directories
home_dir = '/Users/Toby/Downloads/bao/';
input_dir = 'road_construction/rc_all_data/data/data_pilot_preprocessed/';
map_dir = 'road_construction/map/active_map/';
        
fname = fullfile(home_dir,map_dir,'basic_map_48_all4');
basic_map = jsondecode(fileread(fname));

subs = [1,2,4];

for s = 1:length(subs)
    
    sub = subs(s);  
    sub_data = readtable(fullfile(home_dir,input_dir,join(['mod_ibs_preprocess_sub_',int2str(sub),'.csv'])));

    fname = fullfile(home_dir,input_dir,join(['n_repeat_',int2str(sub)]));
    repeats = jsondecode(fileread(fname));
    
    [x,fval] = bads(@ibs_ll,x0,lb,ub,plb,pub,nonbcon,options,[sub_data,repeats,basic_map]);
end


function y = ibs_ll(x,sub_data,repeats,basic_map)
    %ROSENBROCKS Rosenbrock's 'banana' function in any dimension.

    y = py.ibs_bads.ibs_interface(x(1), x(2), x(3), x(4), x(5), x(6), x(7) ,sub_data,repeats,basic_map);
    disp(y);
end
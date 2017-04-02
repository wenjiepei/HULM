function test_err = UCR_experiment(dataset_name, if_lambda, lambda, no_hidden, optimized_method)
%UCR_EXPERIMENT Runs experiment on UCR set
%
% (C) Wenjie Pei, 2016
% Delft University of Technology

clc;
% Process inputs
% type = 1: for determining lambda
% type = 0: for cross validation of performance
if ~exist('dataset_name', 'var') || isempty('dataset_name') %%% 'SwedishLeaf' or 'synthetic_control'
    dataset_name = 'SwedishLeaf';
end
if ~exist('if_lambda', 'var') || isempty(if_lambda)
    if_lambda = 0;
end
if ~exist('eta', 'var') || isempty(eta)
    eta = 5e-3;
end
if ~exist('no_hidden', 'var') || isempty(no_hidden)
    no_hidden = 100;
end
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 1;
end

if ~exist('optimized_method', 'var') || isempty(optimized_method)
    optimized_method = 'SGD';
end

lambdas = [0 1e-5 2e-5 5e-5 .0001 .001 .005 .01 .1 1];
lambda = lambdas(lambda);
load([ 'data' filesep 'UCR_data', filesep, 'UCR_TS_Archive_2015' filesep dataset_name filesep  dataset_name '_new.mat']);
out_dir = ['result' filesep dataset_name filesep num2str(no_hidden)];
if ~exist(out_dir)
    mkdir(out_dir);
end
diary([out_dir filesep 'output_log_no_hidden_' num2str(no_hidden) '_if_lambda_' num2str(if_lambda)...
    '_lambda_' num2str(lambda) '.txt']);
diary on;
labels_n = max(train_T);
disp(['size of training dataset: ', num2str(length(train_X))]);
disp(['size of test dataset: ', num2str(length(test_X))]);
if if_lambda == 1
    disp('cross validation for lambda....');
    disp(['lambda: ' num2str(lambda)]);
    
    %%% use 0.7 in train_X as the train data for validation of lambda and
    %%% left as test data
    randp = randperm(length(train_X));
    train_s = floor(length(train_X) * 0.6);
    train_ind = randp(1:train_s);
    test_ind = randp(train_s+1:end);
    train_X_new = train_X(train_ind);
    train_T_new = train_T(train_ind);
    test_X  = train_X(test_ind);
    test_T  = train_T(test_ind);
    train_X = train_X_new;
    train_T = train_T_new;
else
    disp('cross validation for final performance....');
    disp(['lambda: ' num2str(lambda)]);

end
disp(['trainning set: ' num2str(length(train_T)), '    test set: ' num2str(length(test_T))]);

index_fold = 1;
max_iter = 400;
% Perform training
if strcmp(optimized_method, 'SGD')
    disp('perform SGD...');
    annealing = 0.5;
    eta = 1e-2;
    batch_size = 1;
    decay_threshold = max(1, round(2000/length(train_T)));
    decay_threshold = 6;
    decay_threshold = min(30, decay_threshold);
    decay_threshold = 4;
    stop_threshold = 3*decay_threshold;
    model = train_hidden_unit_logisitic_sgd2(train_X, train_T, test_X, test_T, ...
        lambda, max_iter, eta, batch_size, no_hidden, index_fold, if_lambda, dataset_name, annealing, decay_threshold, stop_threshold);
elseif strcmp(optimized_method, 'LBFGS')
    disp('perform LBFGS...');
    model = train_hidden_unit_logisitic_lbfgs(train_X, train_T, test_X, test_T, ...
        lambda, max_iter, no_hidden, if_lambda, dataset_name);
else
    error('sorry, does not support such optimized method...');
end

if ~exist(['result' filesep dataset_name filesep num2str(no_hidden)])
    mkdir(['result' filesep dataset_name filesep num2str(no_hidden)]);
end
if if_lambda==1
    fid = fopen(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'accuracy_result_lambda_' num2str(lambda) '.txt'], 'w');
else
    fid = fopen(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'accuracy_result_cross_performance_' num2str(lambda) '_' num2str(index_fold) '.txt'], 'w');
end
% Measure training error
trn_err = measure_error(train_X, train_T, model);
disp(['Classification error (training set): ' num2str(trn_err)]);
fprintf(fid, '%s\n', ['Classification error (training set): ' num2str(trn_err)]);

% Perform prediction on test set
test_err = measure_error(test_X, test_T, model);
disp(['Classification error (test set): ' num2str(test_err)]);
fprintf(fid, '%s\n', ['Classification error (test set): ' num2str(test_err)]);
fclose(fid);



end





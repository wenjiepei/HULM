function test_err = banana_experiment(if_lambda, lambda, no_hidden, optimized_method, sample_size)
%banana_experiment Runs experiment on banana data set
%
% (C) Wenjie Pei, 2016
% Delft University of Technology

clc;
% Process inputs
% type = 1: for determining lambda
% type = 0: for cross validation of performance
if ~exist('if_lambda', 'var') || isempty(if_lambda)
    if_lambda = 0;
end
if ~exist('eta', 'var') || isempty(eta)
    eta = 5e-3;
end
if ~exist('no_hidden', 'var') || isempty(no_hidden)
    no_hidden = 3;
end
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 1;
end
if ~exist('overlap_value', 'var') || isempty(overlap_value)
    overlap_value = 3;
end

if ~exist('optimized_method', 'var') || isempty(optimized_method)
    optimized_method = 'SGD';
end
if ~exist('sample_size', 'var') || isempty(sample_size)
    sample_size = 2000;
end


dataset_name = ['banana_' num2str(sample_size)];
load([ '..' filesep 'data' filesep 'banana' filesep, 'banana_' num2str(sample_size) '.mat']);
lambdas = [0 1e-5 .0001 .001 .01 .1 1];
lambda = lambdas(lambda);
out_dir = ['result' filesep dataset_name filesep num2str(no_hidden)];
if ~exist(out_dir)
    mkdir(out_dir);
end
diary([out_dir filesep 'output_log_no_hidden_' num2str(no_hidden) '_if_lambda_' num2str(if_lambda)...
    '_lambda_' num2str(lambda) '.txt']);
diary on;
labels_n = 88;
each_label_number = 100;
if if_lambda == 1
    disp('cross validation for lambda....');
    disp(['lambda: ' num2str(lambda)]);
    disp(['size of dataset: ', num2str(length(new_X))]);
    

    % Split into training and test set 
    train_size = length(train_X);
    randp = randperm(train_size);
    traini = floor(0.7*length(randp));
    train_ind = randp(1:traini);
    test_ind = randp(traini+1:end);
  
    train_X = train_X(train_ind);
    train_T = train_T(train_ind);
    test_X  = train_X(test_ind);
    test_T  = train_T(test_ind);
else
    disp('cross validation for final performance....');
    disp(['lambda: ' num2str(lambda)]);

end
disp(['trainning set: ' num2str(length(train_T)), '    test set: ' num2str(length(test_T))]);

index_fold = 1;
max_iter = 300;
% Perform training
if strcmp(optimized_method, 'SGD')
    disp('perform SGD...');
    annealing = 0.5;
    eta = 1e-1;
    batch_size = 1;
    decay_threshold = 8;
    stop_threshold = 24;
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
[trn_err, train_results] = measure_error(train_X, train_T, model);
disp(['Classification error (training set): ' num2str(trn_err)]);
fprintf(fid, '%s\n', ['Classification error (training set): ' num2str(trn_err)]);

% Perform prediction on test set
[test_err, test_results] = measure_error(test_X, test_T, model);
disp(['Classification error (test set): ' num2str(test_err)]);
fprintf(fid, '%s\n', ['Classification error (test set): ' num2str(test_err)]);
fclose(fid);
 
save(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'classification_results_lambda_' num2str(lambda) '.mat'], 'train_results', 'test_results');



end

function [overlap_feature] = get_overlap_feature(feat, times)

T = size(feat, 2);
D = size(feat, 1);
overlap_feature = zeros(D*times, T-times+1);
for i=1:size(overlap_feature, 2)
    for d = 1:times
        overlap_feature(D*(d-1)+1:D*d, i) = [feat(:, i+d-1)];
    end
    
end

end




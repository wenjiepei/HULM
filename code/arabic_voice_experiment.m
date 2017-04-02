function test_err = arabic_voice_experiment(if_lambda, lambda, no_hidden, optimized_method, window, ifperm)
%arabic_voice_EXPERIMENT Runs experiment on arabic data set to classify the
%subject categories
%
% the data is classified by the (88) subjects, each subjects contains 100
% samples from 10 digits. 
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
    no_hidden = 100;
end
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 1;
end
if ~exist('overlap_value', 'var') || isempty(overlap_value)
    overlap_value = 3;
end

if ~exist('optimized_method', 'var') || isempty(optimized_method) %% 'SGD' or 'LBFGS'
    optimized_method = 'SGD';
end

if ~exist('window', 'var') || isempty(window)
    window = 3;
end
if ~exist('ifperm', 'var') || isempty(ifperm)
    ifperm = 1;
end

dataset_name = ['arabic_voice_window_' num2str(window) '_ifperm_' num2str(ifperm)];
lambdas = [0 1e-5 .0001 .001 .01 .1 1];
lambda = lambdas(lambda);
load([ '..' filesep 'data' filesep 'arabic_voice', filesep, 'arabic_voice_window_' num2str(window) '_ifperm_' num2str(ifperm) '.mat']);
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
    % to speedup the experiments, we only use first 20 signatures for each subjects as training set and 30-40
    % signatures as validation data. 
    
    pre = 0;
    train_ind = [];
    test_ind = [];
    training_size = 20;
    test_size = 10;
    for i=1:20
        train_ind = [train_ind pre+1:pre+training_size];
        test_ind = [test_ind pre+training_size+1:pre+training_size+test_size];
        pre = pre + each_label_number;
    end
    train_X = new_X(train_ind);
    train_T = new_labels(train_ind);
    test_X  = new_X(test_ind);
    test_T  = new_labels(test_ind);
else
    disp('cross validation for final performance....');
    disp(['lambda: ' num2str(lambda)]);
    
    pre = 0;
    train_ind = [];
    test_ind = [];
    training_size = 75;
    test_size = 25;
    for i=1:labels_n
        train_ind = [train_ind pre+1:pre+training_size];
        test_ind = [test_ind pre+training_size+1:pre+training_size+test_size];
        pre = pre + each_label_number;
    end
    train_X = new_X(train_ind);
    train_T = new_labels(train_ind);
    test_X  = new_X(test_ind);
    test_T  = new_labels(test_ind);

end
disp(['trainning set: ' num2str(length(train_T)), '    test set: ' num2str(length(test_T))]);

index_fold = 1;
max_iter = 300;
% Perform training
if strcmp(optimized_method, 'SGD')
    disp('perform SGD...');
    annealing = 0.5;
    eta = 5e-3;
    batch_size = 1;
    decay_threshold = 3;
    stop_threshold = 9;
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
trn_err = measure_training_error(train_X, train_T, model);
disp(['Classification error (training set): ' num2str(trn_err)]);
fprintf(fid, '%s\n', ['Classification error (training set): ' num2str(trn_err)]);

% Perform prediction on test set
test_err = measure_test_error(test_X, test_T, model);
disp(['Classification error (test set): ' num2str(test_err)]);
fprintf(fid, '%s\n', ['Classification error (test set): ' num2str(test_err)]);
fclose(fid);



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

function generate_randomperm_lambda()
%generate random perm
load(['data' filesep 'arabic' filesep 'training_set_arabic']);
labels = train_labels;
X = training_set;
labels_n = 10;
each_label_number = zeros(1, labels_n);
for i=1:length(labels)
    each_label_number( labels(i) ) = each_label_number( labels(i) ) + 1;
end
labels_x = cell(labels_n, 1);
for i=1:length(labels)
    labels_x{labels(i), 1} = [labels_x{labels(i)}, i];
end

new_X = cell(1, length(X));
new_labels = zeros(1, length(X));
indt = 1;
for i=1:labels_n
    rperm = randperm(each_label_number(i));
    for j=1:each_label_number(i)
        new_X{1, indt} = X{labels_x{i,1}(rperm(j))};
        new_labels(1, indt) = i;
        indt = indt + 1;
    end
end

save(['data' filesep 'arabic' filesep 'randomperm_arabic_lambda.mat'], 'new_X', 'new_labels', 'labels_n', 'each_label_number');
diary off;
end



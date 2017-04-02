function err = arabic_experiment(if_lambda, lambda, no_hidden, optimized_method)
%ARABIC_EXPERIMENT Runs experiment on arabic spoken data set to classify
%the digit categories
%
% The function returns the misclassified error of the experiment in err.
%
%
% (C) Wenjie Pei, 2014
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
% lambda is [1,2,3,4,5], which is index of lambda_set [0 .001 .01 .1 1];
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 1;
end
if ~exist('combined_value', 'var') || isempty(combined_value)
    combined_value = 3;
end

if ~exist('optimized_method', 'var') || isempty(optimized_method) %% 'SGD' or 'LBFGS'
    optimized_method = 'SGD';
end

dataset_name = 'arabic';
lambdas = [0 .0001 .001 .01 .1 .5 1 2];
lambda = lambdas(lambda);
% % just implement once to generate randomperm data
% generate_randomperm_lambda();

if if_lambda == 1
    disp('cross validation for lambda....');
    load(['data' filesep dataset_name, filesep, 'randomperm_arabic_lambda.mat']);
    disp(['lambda: ' num2str(lambda)]);
    disp(['size of dataset: ', num2str(length(new_X))]);

    % Split into training and test set
    err = 0;
    no_folds = 10;
    if if_lambda == 1
        index_fold = 10;
    else
        disp(['Fold ' num2str(index_fold) ' of ' num2str(no_folds) '...']);
    end
    
    pre = 0;
    train_ind = [];
    test_ind = [];
    for i=1:labels_n
        fold_size = floor(each_label_number(i) ./ no_folds);
        ind = (index_fold-1) * fold_size + 1;
        train_ind = [train_ind pre+1:pre+ind-1 pre+ind+fold_size:pre+each_label_number(i)];
        test_ind = [test_ind pre+ind:pre+ind+fold_size-1];
        pre = pre + each_label_number(i);
    end
    train_X = new_X(train_ind);
    train_T = new_labels(train_ind);
    test_X  = new_X(test_ind);
    test_T  = new_labels(test_ind);
else
    disp('cross validation for final performance....');
    load(['data' filesep 'arabic' filesep 'training_set_arabic.mat']);
    load(['data' filesep 'arabic' filesep 'test_set_arabic.mat']);
    disp(['lambda: ' num2str(lambda)]);
    
    % combine feature
    for i=1:length(training_set)
        training_set{1, i} = get_overlap_feature(training_set{1, i}, combined_value);
    end
    for i=1:length(test_set)
        test_set{1, i} = get_overlap_feature(test_set{1, i}, combined_value);
    end
    
    train_X = training_set;
    train_T = train_labels;
    test_X  = test_set;
    test_T  = test_labels;

end
disp(['trainning set: ' num2str(length(train_T)), '    test set: ' num2str(length(test_T))]);
index_fold = 1;
max_iter = 350;
% Perform training
if strcmp(optimized_method, 'SGD')
    disp('perform SGD...');
    annealing = 0.5;
    eta = 1e-3;
    batch_size = 5;
    model = train_hidden_unit_logisitic_sgd2(train_X, train_T, test_X, test_T, ...
        lambda, max_iter, eta, batch_size, no_hidden, index_fold, if_lambda, dataset_name, annealing);
elseif strcmp(optimized_method, 'LBFGS')
    disp('perform LBFGS...');
    model = train_hidden_unit_logisitic_lbfgs(train_X, train_T, test_X, test_T, ...
        lambda, max_iter, no_hidden, if_lambda, dataset_name);
else
    error('sorry, does not support such optimized method...');
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

function [combined_feature] = get_combined_feature(feat, times)
T = size(feat, 2);
D = size(feat, 1);
combined_feature = zeros(D*times, floor(T/times));
for i=1:size(combined_feature, 2)
    for d = 1:times
        combined_feature(D*(d-1)+1:D*d, i) = [feat(:, times*(i-1)+d)];
    end
    
end
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

end



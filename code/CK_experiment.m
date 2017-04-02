function err = CK_experiment(if_lambda, lambda, no_hidden, index_fold)
%CK_EXPERIMENT Runs experiment on CK data set
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
if ~exist('index_fold', 'var') || isempty(index_fold)
    index_fold = 1;
end
if ~exist('no_hidden', 'var') || isempty(no_hidden)
    no_hidden = 100;
end
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 1;
end
if ~exist('eta', 'var') || isempty(eta)
    eta = 2e-3;
end

dataset_name = 'CK';
 lambdas = [0 .001 .01 .1 1];
 lambda = lambdas(lambda);
% % just implement once to generate randomperm data
%  generate_randomperm_CK();

% Determine the value of lambda based on cross-validation

%disp('Determining lambda using cross-validation...'); %% the selected value of lambda is 0

if if_lambda == 1
    disp('cross validation for lambda....');
    load(['data' filesep dataset_name, filesep, 'randomperm_CK_lambda.mat']);
    disp(['lambda: ' num2str(lambda)]);
else
    disp('cross validation for final performance....');
    load(['data' filesep, dataset_name, filesep, 'randomperm_CK.mat']);
    disp(['index_fold: ' num2str(index_fold)]);
    disp(['lambda: ' num2str(lambda)]);
end

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

disp(['trainning set: ' num2str(length(train_T)), '    test set: ' num2str(length(test_T))]);

iter = 200;
% Perform predictions 
model = train_hidden_unit_logisitic_sgd(train_X, train_T, test_X, test_T, ...
    lambda, iter, eta, 1, no_hidden, index_fold, if_lambda, dataset_name);

% Measure training error
if if_lambda==1
    fid = fopen(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'accuracy_result_lambda_' num2str(lambda) '.txt'], 'w');
else
    fid = fopen(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'accuracy_result_cross_performance_' num2str(index_fold) '.txt'], 'w');
end

tot = 0; trn_err = 0;
tot = tot + length(train_T);
for i=1:length(train_X)
    if (inference_hidden_unit_logistic_optimized(train_X{i}, model) ~= train_T(i))
        trn_err = trn_err + 1;
    end
end
disp(['Classification error (training set): ' num2str(trn_err / tot)]);
fprintf(fid, '%s\n', ['Classification error (training set): ' num2str(trn_err / tot)]);


% Perform prediction on test set

tot = 0; test_err = 0;
tot = tot + length(test_T);
for i=1:length(test_X)
    if (inference_hidden_unit_logistic_optimized(test_X{i}, model) ~= test_T(i))
        test_err = test_err + 1;
    end
end
disp(['Classification error (test set): ' num2str(test_err / tot)]);
err = test_err / tot;
fprintf(fid, '%s\n', ['Classification error (test set): ' num2str(err)]);

fclose(fid);

end


function generate_randomperm_CK()

load(['data' filesep 'CK' filesep 'cohn_kanade.mat']);
X = X';
labels = emotions;

% Remove data for which we don't have labels
X(labels == -1) = [];
labels(labels == -1) = [];
[foo, bar, labels] = unique(labels);
labels = labels';

labels_n = 7;
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
save(['data' filesep 'CK' filesep 'randomperm_CK_lambda.mat'], 'new_X', 'new_labels', 'labels_n', 'each_label_number');

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


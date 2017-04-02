function err = Action3D_experiment(if_lambda, lambda, no_hidden)
%ACTION3D_EXPERIMENT Runs experiment on Action3D data set
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
    eta = 1e-3;
end
if ~exist('no_hidden', 'var') || isempty(no_hidden)
    no_hidden = 100;
end
% lambda is [1,2,3,4,5], which is index of lambda_set [0 .001 .01 .1 1];
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 1;
end

dataset_name = 'Action3D';
lambdas = [0 .001 .01 .1 1];
lambda = lambdas(lambda);

if if_lambda == 1
    disp('cross validation for lambda....');
    disp(['lambda: ' num2str(lambda)]);
    
    % Split into training and test set
    err = 0;
    load(['data' filesep 'Action3D' filesep 'joint_feat_coordinate.mat']);
    X = feat;
    
    K = 20;
    train_ind = [];
    test_ind = [];
    testActors = [1 3 5 7 9];
    i = 1;
    true_i = 0;
    for a = 1:20
        for j = 1:10
            for e = 1:3
                if if_contain(i)==0
                    i = i+1;
                    continue;
                end
                true_i = true_i + 1;
                if ~isempty(find(testActors == j))
                    test_ind = [test_ind, true_i];
                else
                    train_ind = [train_ind, true_i];
                end
                i = i + 1;
                
            end
        end
    end
    
    train_X = X(train_ind);
    train_T = labels(train_ind);
    test_X  = X(test_ind);
    test_T  = labels(test_ind);
    
else
    disp('cross validation for final performance....');
    disp(['lambda: ' num2str(lambda)]);
    
    load(['data' filesep 'Action3D' filesep 'joint_feat_coordinate.mat']);
    X = feat;
    
    K = 20;
    train_ind = [];
    test_ind = [];
    testActors = [6 7 8 9 10];
    i = 1;
    true_i = 0;
    for a = 1:20
        for j = 1:10
            for e = 1:3
                if if_contain(i)==0
                    i = i+1;
                    continue;
                end
                true_i = true_i + 1;
                if ~isempty(find(testActors == j))
                    test_ind = [test_ind, true_i];
                else
                    train_ind = [train_ind, true_i];
                end
                i = i + 1;
                
            end
        end
    end
    train_X = X(train_ind);
    train_T = labels(train_ind);
    test_X  = X(test_ind);
    test_T  = labels(test_ind);
    
end
disp(['trainning set: ' num2str(length(train_T)), '    test set: ' num2str(length(test_T))]);
iter = 400;
% Perform predictions
index_fold = 1;
annealing = 0.998;
disp(['total iter: ' num2str(iter)]);
model = train_hidden_unit_logisitic_sgd(train_X, train_T, test_X, test_T, ...
    lambda, iter, eta, 1, no_hidden, index_fold, if_lambda, dataset_name, annealing);

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



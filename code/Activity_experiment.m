function err = Activity_experiment(if_lambda, lambda, no_hidden, if_direct_test)
%Activity_experiment Runs experiment on activity data set
%
% The function returns the misclassified error of the experiment in err.
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
if ~exist('if_direct_test', 'var') || isempty(if_direct_test)
    if_direct_test = false;
end

dataset_name = 'Activity';
disp(['Experiments for ' dataset_name]);

lambdas = [0 .001 .01 .1 1];
lambda = lambdas(lambda);

if if_lambda == 1
    disp('cross validation for lambda....');
    disp(['lambda: ' num2str(lambda)]);
    
    % Split into training and test set
    load(['data' filesep 'Activity' filesep 'joint3D_feature_noFFT.mat']);
    X = Joint3D_feature;
    
    K = 16;
    train_ind = [];
    test_ind = [];
    testActors = [1 3 5 7 9];
    true_i = 0;
    for a = 1:16
        for j = 1:10
            for e = 1:2
                true_i = true_i + 1;
                if (~isempty(find(testActors == j)))
                    test_ind = [test_ind, true_i];
                else
                    train_ind = [train_ind, true_i];
                end
                
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
    
    % Split into training and test set
    load(['data' filesep 'Activity' filesep 'joint3D_feature_noFFT.mat']);
    X = Joint3D_feature;
    
    K = 16;
    train_ind = [];
    test_ind = [];
    testActors = [1 2 3 4 5];
    testClass = [1:16];
    true_i = 0;
    for a = 1:16
        for j = 1:10
            for e = 1:2
                true_i = true_i + 1;
                if ~isempty(find(testActors == j))
                    test_ind = [test_ind, true_i];
                else
                    train_ind = [train_ind, true_i];
                end
                
            end
        end
    end
    
    train_X = X(train_ind);
    train_T = labels(train_ind);
    test_X  = X(test_ind);
    test_T  = labels(test_ind);
    if length(testClass) == 1
        one_set = find(train_T==testClass(1));
        zero_set = setdiff([1:length(train_T)], one_set);
        train_T(one_set) = 1;
        train_T(zero_set) = 2;
        one_set = find(test_T==testClass(1));
        zero_set = setdiff([1:length(test_T)], one_set);
        test_T(one_set) = 1;
        test_T(zero_set) = 2;
    end
end
disp(['trainning set: ' num2str(length(train_T)), '    test set: ' num2str(length(test_T))]);
index_fold = 1;
iter = 300;
% Perform predictions
if ~if_direct_test
    annealing = 0.997;
    model = train_hidden_unit_logisitic_sgd(train_X, train_T, test_X, test_T, ...
        lambda, iter, eta, 1, no_hidden, index_fold, if_lambda, dataset_name, annealing);
else
    load(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'model_temp_cross_validation_',num2str(lambda) '_' num2str(index_fold), '.mat']);
end

% Measure training error
if if_lambda==1
    fid = fopen(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'accuracy_result_lambda_' num2str(lambda) '.txt'], 'w');
else
    fid = fopen(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'accuracy_result_cross_performance_' num2str(lambda) '_' num2str(index_fold) '.txt'], 'w');
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
test_prob = zeros(length(test_X), 1);
for i=1:length(test_X)
    [pred_label, pred_prob] = inference_hidden_unit_logistic_optimized(test_X{i}, model);
    if ( pred_label ~= test_T(i))
        test_err = test_err + 1;
    end
    if pred_label == 1
        test_prob(i) = pred_prob;
    else
        test_prob(i) = 1 - pred_prob;
    end
end
disp(['Classification error (test set): ' num2str(test_err / tot)]);
err = test_err / tot;
fprintf(fid, '%s\n', ['Classification error (test set): ' num2str(err)]);
if length(testClass == 1)
    auc = get_AUC(test_T, 1,  test_prob);
    disp(['auc: ', num2str(auc)]);
    EER = get_EER(test_T, 1, test_prob);
    disp(['EER: ', num2str(EER)]);
end

fclose(fid);

end




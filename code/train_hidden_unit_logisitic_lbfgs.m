function [ model ] = train_hidden_unit_logisitic_lbfgs(train_X, train_T, test_X, test_T, ...
    lambda, max_iter, no_hidden, if_lambda, dataset_name)
% LBFGS
% (C) Wenjie Pei, 2016
% Delft University of Technology

addpath(genpath(['minFunc_2009' filesep]));

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end
if ~exist('max_iter', 'var') || isempty(max_iter)
    max_iter = 10;
end

if ~exist(['result' filesep dataset_name filesep num2str(no_hidden)])
    mkdir(['result' filesep dataset_name filesep num2str(no_hidden)]);
end
diary(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'output_if_lambda_' num2str(if_lambda)...
    '_lambda_' num2str(lambda) '.txt']);
diary on;

averaging = true;
burnin_iter = round(2 * max_iter / 3);

% Compute number of features / dimensionality and number of labels
% size of Trainning set
N = length(train_X);
D = size(train_X{1}, 1);
% @K is the class number
K = 0;
for i=1:length(train_T)
    K = max(K, max(train_T(i)));
end

disp(['N: ', num2str(N), ' D: ', num2str(D), ' K: ', num2str(K)]);

save_file_dir = ['result' filesep dataset_name filesep num2str(no_hidden)];
if ~exist(save_file_dir)
    mkdir(save_file_dir);
end

save_model_dir = [save_file_dir filesep 'lbfgs_if_lambda_' num2str(if_lambda) '_lambda_' num2str(lambda) '.mat'];
if exist(save_model_dir)
    disp('continue training from last saved model...');
    load(save_model_dir);
else
    % Initialize model (all parameters are in the log-domain)
    model.pi  = zeros(no_hidden, 1);
    model.tau = zeros(no_hidden, 1);
    model.A   = randn(no_hidden, 1) * .1;
    model.W = randn(no_hidden, D) * .01;
    model.b_bias = zeros(no_hidden, 1); % equivalent to 'b' in the equation
    model.V = randn(no_hidden, K) * .01;
    model.c_bias = zeros(1, K); % equivalent to 'c' in the equation
end

% Initialize parameter vector
x = [model.pi(:); model.tau(:); model.A(:); model.W(:); model.V(:); model.b_bias(:); model.c_bias(:)];

% LBFGS to find the weights
display('Training...');
options.dataset_name = dataset_name;
options.LS=0;
options.TolFun=1e-2;
options.TolX=1e-2;
options.Method='lbfgs';
options.Display='on';
options.MaxIter=max_iter;
options.DerivativeCheck='off';

x = minFunc(@hidden_unit_logistic_grad_optimized, x, options, train_X, train_T, model, lambda);

% Decode current solution
ind = 1;
model.pi  = reshape(x(ind:ind + numel(model.pi)  - 1), size(model.pi));  ind = ind + numel(model.pi);
model.tau = reshape(x(ind:ind + numel(model.tau) - 1), size(model.tau)); ind = ind + numel(model.tau);
model.A   = reshape(x(ind:ind + numel(model.A)   - 1), size(model.A));   ind = ind + numel(model.A);
model.W   = reshape(x(ind:ind + numel(model.W)   - 1), size(model.W));   ind = ind + numel(model.W);
model.V      = reshape(x(ind:ind + numel(model.V)      - 1), size(model.V));   ind = ind + numel(model.V);
model.b_bias    = reshape(x(ind:ind + numel(model.b_bias)    - 1), size(model.b_bias)); ind = ind + numel(model.b_bias);
model.c_bias = reshape(x(ind:ind + numel(model.c_bias) - 1), size(model.c_bias));

save(save_model_dir, 'model');
% final evaluate on the training and test dataset
trn_err = measure_training_error(train_X, train_T, model);
test_err = measure_test_error(test_X, test_T, model);
disp(['training error: ', num2str(trn_err)]);
disp(['test error: ', num2str(test_err)]);

end


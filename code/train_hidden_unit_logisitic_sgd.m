function [ model ] = train_hidden_unit_logisitic_sgd(train_X, train_T, test_X, test_T, ...
    lambda, max_iter, eta, batch_size, no_hidden, ind_cross, if_lambda, dataset_name, annealing)
%TRAIN_HIDDEN_UNIT_LOGISITC_SGD trains a hidden-unit logisitc model using
%stochastic gradient descent
%   model = train_hidden_unit_logistic_sgd(train_X, train_T, type, lambda, max_iter, eta, batch_size, no_hidden)
%
% (C) Wenjie Pei, 2014
% Delft University of Technology



if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end
if ~exist('max_iter', 'var') || isempty(max_iter)
    max_iter = 10;
end
if ~exist('eta', 'var') || isempty(eta)
    eta = 1e-5;
end
if ~exist('batch_size', 'var') || isempty(batch_size)
    batch_size = 1;
end
if ~exist('annealing', 'var') || isempty(annealing)
    annealing = 0.995;
end

annealing = 0.99;
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

disp(['D: ', num2str(D), ' K: ', num2str(K)]);

iter = 1;
err = 0;

if ~exist(['result' filesep dataset_name filesep num2str(no_hidden)])
    mkdir(['result' filesep dataset_name filesep num2str(no_hidden)]);
end

if (if_lambda==1 && exist(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'model_temp_lambda_', num2str(lambda), '.mat']) ||...
        if_lambda==0 && exist(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'model_temp_cross_validation_', num2str(lambda) '_' num2str(ind_cross), '.mat']))
    if if_lambda==1
        load(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'model_temp_lambda_', num2str(lambda), '.mat']);
    else
        load(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'model_temp_cross_validation_',num2str(lambda) '_' num2str(ind_cross), '.mat']);
    end

    iter = iter + 1;
% eta = 0.0004;
    disp(['iter: ', num2str(iter), '   eta: ', num2str(eta)]);
    if length(err) < max_iter
        errt = zeros(max_iter, 1);
        errt(1:length(err), 1) = err;
        err = errt;
    end
    ii = iter - burnin_iter;
    
else
    % Initialize model (all parameters are in the log-domain)
    model.pi  = zeros(no_hidden, 1);
    model.tau = zeros(no_hidden, 1);
    model.A   = randn(no_hidden, 1) * .1;
    model.W = randn(no_hidden, D) * .001;
    model.b_bias = zeros(no_hidden, 1); % equivalent to 'b' in the equation
    model.V = randn(no_hidden, K) * .001;
    model.c_bias = zeros(1, K); % equivalent to 'c' in the equation
    err = zeros(max_iter, 1);
end

% Initialize parameter vector
x = [model.pi(:); model.tau(:); model.A(:); model.W(:); model.V(:); model.b_bias(:); model.c_bias(:)];
mean_x = x;
   
% Perform minimization using stochastic gradient descent
disp('Performing optimization using SGD...');

if if_lambda==1
    fid = fopen(['result', filesep, dataset_name, filesep, num2str(no_hidden) filesep 'resultC_lambda_', num2str(lambda), '.txt'], 'a');
    fidd = fopen(['result' filesep, dataset_name, filesep,  num2str(no_hidden) filesep 'accuracy_result_mid_lambda_' num2str(lambda) '.txt'], 'a');
else
    fid = fopen(['result', filesep, dataset_name, filesep, num2str(no_hidden) filesep 'resultC_cross_validation_', num2str(lambda) '_' num2str(ind_cross), '.txt'], 'a');
    fidd = fopen(['result' filesep, dataset_name, filesep, num2str(no_hidden) filesep  'accuracy_result_mid_cross_validation_' num2str(lambda) '_' num2str(ind_cross) '.txt'], 'a');
end

step_statistic = 50;
step_iter = 10;
best_err = 1;
better_time = 0;

for iter=iter:max_iter
    disp(['start iteration ' num2str(iter), ' eta: ', num2str(eta)]);
    start_train = clock;
    t2_train_loop = start_train;
    old_x = x;
    rand_ind = randperm(N);
    eta = eta * annealing;
    
    last_step = 0;
    % Loop over all time series
    for i=1:batch_size:N
        t1_train_loop = t2_train_loop;
        
        % Perform gradient update for single time series
        cur_ind = rand_ind(i:min(i + batch_size - 1, N));
        %             [C1, ~, x1] = hidden_unit_logistic_grad(x1, train_X(cur_ind), train_T(cur_ind), model1, lambda, eta / batch_size);
        [C, dC] = hidden_unit_logistic_grad_optimized(x, train_X(cur_ind), train_T(cur_ind), model, lambda);
        %update the parameters
        x = x - eta / batch_size * dC;
        err(iter) = err(iter) + C;
        last_step = last_step + C;
        if mod(i, step_statistic) == 0
            t2_train_loop = clock;
            disp(['time for training intances from ', num2str(i-step_statistic), ' to ', num2str(i),' : ', num2str(etime(t2_train_loop, t1_train_loop))]);
            disp(['average C: ', num2str(err(iter) / (i)), '    last ' num2str(step_statistic) ' average C: ', num2str(last_step / step_statistic)]);
            last_step = 0;
            %                 fprintf(fid, '%s\n', ['C: ' num2str(C)]);
        end
        
    end
    err(iter) = err(iter) / N;
    disp(['Iteration ' num2str(iter) ' of ' num2str(max_iter) ': error is: ' num2str(err(iter))]);
    if iter == 1
        best_err = err(iter);
    else
        if best_err < err(iter)
            better_time = better_time + 1;
        else
            better_time = 0;
            best_err = min(err(iter), best_err);
        end
    end
    disp(['better time: ', num2str(better_time)]);
    if better_time > 15
        disp('training finished: better time > 15');
        break;
    end
    
    end_train = clock;
    disp(['Iteration ' num2str(iter) ' of ' num2str(max_iter) ': elapsed time is: ' num2str(etime(end_train, start_train))]);
    fprintf(fid, '%s\n', ['Iteration ' num2str(iter) ' of ' num2str(max_iter) ': error is: ' num2str(err(iter)), '  with eta: ', num2str(eta)]);
    
    % Decode current solution
    ind = 1;
    model.pi  = reshape(x(ind:ind + numel(model.pi)  - 1), size(model.pi));  ind = ind + numel(model.pi);
    model.tau = reshape(x(ind:ind + numel(model.tau) - 1), size(model.tau)); ind = ind + numel(model.tau);
    model.A   = reshape(x(ind:ind + numel(model.A)   - 1), size(model.A));   ind = ind + numel(model.A);
    model.W   = reshape(x(ind:ind + numel(model.W)   - 1), size(model.W));   ind = ind + numel(model.W);
    model.V      = reshape(x(ind:ind + numel(model.V)      - 1), size(model.V));   ind = ind + numel(model.V);
    model.b_bias    = reshape(x(ind:ind + numel(model.b_bias)    - 1), size(model.b_bias)); ind = ind + numel(model.b_bias);
    model.c_bias = reshape(x(ind:ind + numel(model.c_bias) - 1), size(model.c_bias));
    
    is_InF_NaN(model.pi);
    is_InF_NaN(model.tau);
    is_InF_NaN(model.A);
    is_InF_NaN(model.W);
    is_InF_NaN(model.V);
    is_InF_NaN(model.b_bias);
    is_InF_NaN(model.c_bias);
    
    % Perform averaging after certain number of iterations
    if iter < burnin_iter || ~averaging
        ii = 0;
        mean_x = x;
        mean_model = model;
    else
        ii = ii + 1;
        mean_x = ((ii - 1) / ii) .* mean_x + (1 / ii) .* x;
        ind = 1;
    mean_model.pi  = reshape(mean_x(ind:ind + numel(model.pi)  - 1), size(model.pi));  ind = ind + numel(model.pi);
    mean_model.tau = reshape(mean_x(ind:ind + numel(model.tau) - 1), size(model.tau)); ind = ind + numel(model.tau);
    mean_model.A   = reshape(mean_x(ind:ind + numel(model.A)   - 1), size(model.A));   ind = ind + numel(model.A);
    mean_model.W   = reshape(mean_x(ind:ind + numel(model.W)   - 1), size(model.W));   ind = ind + numel(model.W);
    mean_model.V      = reshape(mean_x(ind:ind + numel(model.V)      - 1), size(model.V));   ind = ind + numel(model.V);
    mean_model.b_bias    = reshape(mean_x(ind:ind + numel(model.b_bias)    - 1), size(model.b_bias)); ind = ind + numel(model.b_bias);
    mean_model.c_bias = reshape(mean_x(ind:ind + numel(model.c_bias) - 1), size(model.c_bias));
    end
    model = mean_model;
    if if_lambda==1
        save(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'model_temp_lambda_', num2str(lambda), '.mat'], 'model', 'iter', 'err', 'eta');
    elseif better_time == 0
        save(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'model_temp_cross_validation_',num2str(lambda) '_' num2str(ind_cross), '.mat'], 'model', 'iter', 'err', 'eta');
    end
    
    if (mod(iter, step_iter) == 0)
        avg_err = sum(err(iter-step_iter+1:iter,1)) / step_iter;
        disp(['average err in the past ' num2str(step_iter) ' frames: ', num2str(avg_err)]);
        fprintf(fid,  '%s\n', ['average err in the past ' num2str(step_iter) ' frames: error is: ' num2str(avg_err)]);
    end
    if (mod(iter, step_iter) == 0)
        trn_err = measure_training_error(train_X, train_T, model);
        test_err = measure_test_error(test_X, test_T, model);
        fprintf(fidd, '%s\n', ['Classification error (training set) ' num2str(iter) ': ' num2str(trn_err)]);
        fprintf(fidd, '%s\n', ['Classification error (test set) ' num2str(iter) ': ' num2str(test_err) '  with C: '  num2str(err(iter))]);
        
    end
end


fclose(fid);
fclose(fidd);

end

function trn_err = measure_training_error(train_X, train_T, model)

disp('measuring training error...');
t1 = clock;
tot = 0; trn_err = 0;
tot = tot + length(train_T);
for i=1:length(train_X)
    %         trn_err = trn_err + (inference_hidden_unit_logistic(train_X{i}, model) ~= train_T(i));
    trn_err = trn_err + (inference_hidden_unit_logistic_optimized(train_X{i}, model) ~= train_T(i));
end
trn_err = trn_err / tot;
disp(['Classification error (training set): ' num2str(trn_err)]);
t2 = clock;
disp(['classification time: ', num2str(etime(t2, t1))]);

end

function test_err = measure_test_error(test_X, test_T, model)

disp('measuring test error...');
t1 = clock;
tot = 0; test_err = 0;
tot = tot + length(test_T);
for i=1:length(test_X)
    %         test_err = test_err + (inference_hidden_unit_logistic(test_X{i}, model) ~= test_T(i));
    test_err = test_err + (inference_hidden_unit_logistic_optimized(test_X{i}, model) ~= test_T(i));
end
test_err = test_err / tot;
disp(['Classification error (test set): ' num2str(test_err)]);
t2 = clock;
disp(['classification time: ', num2str(etime(t2, t1))]);

end
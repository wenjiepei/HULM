function [ model ] = train_hidden_unit_logisitic_sgd2(train_X, train_T, test_X, test_T, ...
    lambda, max_iter, eta, batch_size, no_hidden, index_fold, if_lambda, dataset_name, annealing, decay_threshold, stop_threshold, if_save_model)
% decrease the learning rate only when the losss value is not decreased
% anymore
% (C) Wenjie Pei, 2016
% Delft University of Technology



if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end
if ~exist('max_iter', 'var') || isempty(max_iter)
    max_iter = 200;
end
if ~exist('eta', 'var') || isempty(eta)
    eta = 1e-3;
end
if ~exist('batch_size', 'var') || isempty(batch_size)
    batch_size = 5;
end
if ~exist('annealing', 'var') || isempty(annealing)
    annealing = 0.5;
end
if ~exist('decay_threshold', 'var') || isempty(decay_threshold)
    decay_threshold = 5;
end
if ~exist('stop_threshold', 'var') || isempty(stop_threshold)
    stop_threshold = 15;
end
if ~exist('if_save_model', 'var') || isempty(if_save_model)
    if_save_model = true;
end

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

err = 0;

if ~exist(['result' filesep dataset_name filesep num2str(no_hidden)])
    mkdir(['result' filesep dataset_name filesep num2str(no_hidden)]);
end

if (if_lambda==1 && exist(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'sgd_model_temp_lambda_', num2str(lambda), '.mat']) ||...
        if_lambda==0 && exist(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'sgd_model_temp_cross_validation_', num2str(lambda) '_' num2str(index_fold), '.mat']))
    if if_lambda==1
        load(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'sgd_model_temp_lambda_', num2str(lambda), '.mat']);
    else
        load(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'sgd_model_temp_cross_validation_',num2str(lambda) '_' num2str(index_fold), '.mat']);
    end
    iter = iter + 1;
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
    model.A   = randn(no_hidden, 1) * .01;
    model.W = randn(no_hidden, D) * .01;
    model.b_bias = zeros(no_hidden, 1); % equivalent to 'b' in the equation
    model.V = randn(no_hidden, K) * .01;
%     model.A   = rand(no_hidden, 1) * .001;
%     model.W = rand(no_hidden, D) * .001;
%     model.b_bias = rand(no_hidden, 1) * .001; % equivalent to 'b' in the equation
%     model.V = rand(no_hidden, K) * .001;
    model.c_bias = zeros(1, K); % equivalent to 'c' in the equation
    err = rand(max_iter, 1) * .001;
    iter = 1;
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
    fid = fopen(['result', filesep, dataset_name, filesep, num2str(no_hidden) filesep 'resultC_cross_validation_', num2str(lambda) '_' num2str(index_fold), '.txt'], 'a');
    fidd = fopen(['result' filesep, dataset_name, filesep, num2str(no_hidden) filesep  'accuracy_result_mid_cross_validation_' num2str(lambda) '_' num2str(index_fold) '.txt'], 'a');
end

step_statistic = 20;
step_iter = max(1, round(2000/N));
step_iter = 2;
step_iter = min(20, step_iter);
best_err = 1;
better_time = 0;
if_gradient_check = false;
if_initial = true;

for iter=iter:max_iter
    disp(['start iteration ' num2str(iter), ' eta: ', num2str(eta)]);
    start_train = clock;
    t2_train_loop = start_train;
    old_x = x;
    rand_ind = randperm(N);
    
    last_step = 0;
    % Loop over all time series
    for i=1:batch_size:N
        t1_train_loop = t2_train_loop;
        
        % Perform gradient update for single time series
        eind = min(i + batch_size - 1, N);
        cur_ind = rand_ind(i:eind);
        [C, dC] = hidden_unit_logistic_grad_optimized(x, train_X(cur_ind), train_T(cur_ind), model, lambda);
        if if_gradient_check %%% only for batch_size==1
            gradient_check(train_X{cur_ind}, train_T(cur_ind), model, dC);
            error('finish gradient check.');
        end
        
            
        x = x - eta / batch_size * dC;
        err(iter) = err(iter) + C;
        last_step = last_step + C;
        if mod(eind, step_statistic) == 0
            t2_train_loop = clock;
            disp(['time for training intances from ', num2str(eind-step_statistic+1), ' to ', num2str(eind),' : ', num2str(etime(t2_train_loop, t1_train_loop))]);
            disp(['average C: ', num2str(err(iter) / eind), '    last ' num2str(step_statistic) ' average C: ', num2str(last_step / (step_statistic))]);
            last_step = 0;
            %                 fprintf(fid, '%s\n', ['C: ' num2str(C)]);
        end
        
    end
    err(iter) = err(iter) / N;
    disp(['Iteration ' num2str(iter) ' of ' num2str(max_iter) ': error is: ' num2str(err(iter))]);
    if if_initial
        best_err = err(iter);
    else
        if best_err < err(iter)
            better_time = better_time + 1;
        else
            better_time = 0;
            best_err = min(err(iter), best_err);
        end
    end
    if_initial = false;
    disp(['better time: ', num2str(better_time)]);
    if better_time > stop_threshold
        disp(['training finished: better time > ' num2str(stop_threshold)]);
        break;
    end
    if mod(better_time, decay_threshold) == 0 && better_time >= decay_threshold
       eta = eta * annealing; 
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
    if iter < 2 || iter < burnin_iter || ~averaging
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
    if if_save_model
        if if_lambda==1
            save(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'sgd_model_temp_lambda_', num2str(lambda), '.mat'], 'model', 'iter', 'err', 'eta');
        elseif better_time == 0
            save(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'sgd_model_temp_cross_validation_',num2str(lambda) '_' num2str(index_fold), '.mat'], 'model', 'iter', 'err', 'eta');
        end
    end
    
    if (mod(iter, step_iter) == 0)
        avg_err = sum(err(iter-step_iter+1:iter,1)) / step_iter;
        disp(['average err in the past ' num2str(step_iter) ' frames: ', num2str(avg_err)]);
        fprintf(fid,  '%s\n', ['average err in the past ' num2str(step_iter) ' frames: error is: ' num2str(avg_err)]);
    end
    if (mod(iter, step_iter) == 0)
        [trn_err, train_results] = measure_error(train_X, train_T, model);
        [test_err, test_results] = measure_error(test_X, test_T, model);
 
        save(['result' filesep dataset_name filesep num2str(no_hidden) filesep 'classification_results_lambda_' num2str(lambda) '.mat'], 'train_results', 'test_results');
        fprintf(fidd, '%s\n', ['Classification error (training set) ' num2str(iter) ': ' num2str(trn_err)]);
        fprintf(fidd, '%s\n', ['Classification error (test set) ' num2str(iter) ': ' num2str(test_err) '  with C: '  num2str(err(iter))]);
%         if trn_err == 0
%             disp('the training error is 0 !!!');
%             break;
%         end
    end
end


fclose(fid);
fclose(fidd);

end









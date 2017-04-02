function [trn_err, results] = measure_error(X, T, model)

if ~exist('if_save_result', 'var') || isempty(if_save_result)
    if_save_result = false;
end
disp('measuring error...');
t1 = clock;
tot = 0; trn_err = 0;
tot = tot + length(T);
results = zeros(length(X), 2);%% first column is the true_labels, the second column is the predicted label
results(:, 1) = T';
for i=1:length(X)
    results(i, 2) = inference_hidden_unit_logistic_optimized(X{i}, model);
    trn_err = trn_err + (results(i, 2) ~= T(i));
end
trn_err = trn_err / tot;
disp(['Classification error: ' num2str(trn_err)]);
t2 = clock;
disp(['classification time: ', num2str(etime(t2, t1))]);

end
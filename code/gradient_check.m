function gradient_check(X, label, model, dC)

disp('start to check gradient... ');
step = 5e-5;
temp_model = model;
ind = 1;
d_pi  = dC(ind:ind + numel(model.pi)  - 1);  ind = ind + numel(model.pi);
d_tau = dC(ind:ind + numel(model.tau) - 1); ind = ind + numel(model.tau);
d_A   = dC(ind:ind + numel(model.A)   - 1);   ind = ind + numel(model.A);
d_W   = dC(ind:ind + numel(model.W)   - 1);   ind = ind + numel(model.W);
d_V   = dC(ind:ind + numel(model.V)   - 1);   ind = ind + numel(model.V);
d_b_bias = dC(ind:ind + numel(model.b_bias) - 1); ind = ind + numel(model.b_bias);
d_c_bias = dC(ind:ind + numel(model.c_bias) - 1);

% %%% check the gradient of A
% gradient = zeros(numel(model.A),1);
% temp = model.A(:);
% for i = 1:numel(model.A)
%     temp(i) = temp(i) + step;
%     new_A = reshape(temp, size(model.A));
%     temp_model.A = new_A;
%     C2 = get_C(X, label, temp_model);
%     temp(i) = temp(i) - 2*step;
%     new_A = reshape(temp, size(model.A));
%     temp_model.A = new_A;
%     C1 = get_C(X, label, temp_model);
%     temp(i) = temp(i) + step;
%     gradient(i) = (C2-C1) / (2*step);
%     diff = abs(gradient(i) - d_A(i)) / max(abs(gradient(i)), abs(d_A(i)));
%     disp([num2str(i), ':  ' num2str(diff), '  math_g:  ' num2str(gradient(i))  '      dC: ' num2str(d_A(i))]);
% end

% %%% check the gradient of W
% gradient = zeros(numel(model.W),1);
% temp = model.W(:);
% for i = 1:numel(model.W)
%     temp(i) = temp(i) + step;
%     new_W = reshape(temp, size(model.W));
%     temp_model.W = new_W;
%     C2 = get_C(X, label, temp_model);
%     temp(i) = temp(i) - 2*step;
%     new_W = reshape(temp, size(model.W));
%     temp_model.W = new_W;
%     C1 = get_C(X, label, temp_model);
%     temp(i) = temp(i) + step;
%     gradient(i) = (C2-C1) / (2*step);
%     diff = abs(gradient(i) - d_W(i)) / max(abs(gradient(i)), abs(d_W(i)));
%     disp([num2str(i), ':  ' num2str(diff), '  math_g:  ' num2str(gradient(i))  '      dC: ' num2str(d_W(i))]);
% end

% %%% check the gradient of tau
% gradient = zeros(numel(model.tau),1);
% temp = model.tau(:);
% for i = 1:numel(model.tau)
%     temp(i) = temp(i) + step;
%     new_tau = reshape(temp, size(model.tau));
%     temp_model.tau = new_tau;
%     C2 = get_C(X, label, temp_model);
%     temp(i) = temp(i) - 2*step;
%     new_tau = reshape(temp, size(model.tau));
%     temp_model.tau = new_tau;
%     C1 = get_C(X, label, temp_model);
%     temp(i) = temp(i) + step;
%     gradient(i) = (C2-C1) / (2*step);
%     diff = abs(gradient(i) - d_tau(i)) / max(abs(gradient(i)), abs(d_tau(i)));
%     disp([num2str(i), ':  ' num2str(diff), '  math_g:  ' num2str(gradient(i))  '      dC: ' num2str(d_tau(i))]);
% end

% %%% check the gradient of b_bias
% gradient = zeros(numel(model.b_bias),1);
% temp = model.b_bias(:);
% for i = 1:numel(model.b_bias)
%     temp(i) = temp(i) + step;
%     new_b_bias = reshape(temp, size(model.b_bias));
%     temp_model.b_bias = new_b_bias;
%     C2 = get_C(X, label, temp_model);
%     temp(i) = temp(i) - 2*step;
%     new_b_bias = reshape(temp, size(model.b_bias));
%     temp_model.b_bias = new_b_bias;
%     C1 = get_C(X, label, temp_model);
%     temp(i) = temp(i) + step;
%     gradient(i) = (C2-C1) / (2*step);
%     diff = abs(gradient(i) - d_b_bias(i)) / max(abs(gradient(i)), abs(d_b_bias(i)));
%     disp([num2str(i), ':  ' num2str(diff), '  math_g:  ' num2str(gradient(i))  '      dC: ' num2str(d_b_bias(i))]);
% end


% %%% check the gradient of c_bias
% gradient = zeros(numel(model.c_bias),1);
% temp = model.c_bias(:);
% for i = 1:numel(model.c_bias)
%     temp(i) = temp(i) + step;
%     new_c_bias = reshape(temp, size(model.c_bias));
%     temp_model.c_bias = new_c_bias;
%     C2 = get_C(X, label, temp_model);
%     temp(i) = temp(i) - 2*step;
%     new_c_bias = reshape(temp, size(model.c_bias));
%     temp_model.c_bias = new_c_bias;
%     C1 = get_C(X, label, temp_model);
%     temp(i) = temp(i) + step;
%     gradient(i) = (C2-C1) / (2*step);
%     diff = abs(gradient(i) - d_c_bias(i)) / max(abs(gradient(i)), abs(d_c_bias(i)));
%     disp([num2str(i), ':  ' num2str(diff), '  math_g:  ' num2str(gradient(i))  '      dC: ' num2str(d_c_bias(i))]);
% end


%%% check the gradient of pi
gradient = zeros(numel(model.pi),1);
temp = model.pi(:);
for i = 1:numel(model.pi)
    temp(i) = temp(i) + step;
    new_pi = reshape(temp, size(model.pi));
    temp_model.pi = new_pi;
    C2 = get_C(X, label, temp_model);
    temp(i) = temp(i) - 2*step;
    new_pi = reshape(temp, size(model.pi));
    temp_model.pi = new_pi;
    C1 = get_C(X, label, temp_model);
    temp(i) = temp(i) + step;
    gradient(i) = (C2-C1) / (2*step);
    diff = abs(gradient(i) - d_pi(i)) / max(abs(gradient(i)), abs(d_pi(i)));
    disp([num2str(i), ':  ' num2str(diff), '  math_g:  ' num2str(gradient(i))  '      dC: ' num2str(d_pi(i))]);
end




end


function C = get_C(X, t_label, model)
K = length(model.c_bias);
D = size(model.W, 2);
T = size(X, 2);
[~, ~, log_M, ~, ~, ~, mid_psi_result] = forward_backward_hidden_unit_logistic_optimized_cmex(X, t_label, model, 0, 0);
log_M_temp = zeros(K, 1);
log_Xi_vec = cell(K, 1);
for label=1:K
    [~, ~, log_M_temp(label, 1), ~, ~, log_Xi_vec{label, 1}, ~] = forward_backward_hidden_unit_logistic_optimized_cmex(X, label, model, 1, 1, mid_psi_result);
end
max_v = max(log_M_temp);
diff = log_M_temp - max_v;
sum_diff = sum(exp(diff));
log_Z = max_v + log(sum_diff);
C = log_Z - log_M;
end
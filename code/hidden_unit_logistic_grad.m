function [ C, dC, x ]  = hidden_unit_logistic_grad( x, train_X, train_T, model, lambda, eta )
%HIDDEN_UNIT_LOGISTIC_GRAD Computes conditional log-likelihood and gradient
%
%
% Compute negative conditional log-likelihood C and the corresponding
% gradient on the specified training time series (train_X, train_T).
%
% (C) Wenjie Pei, 2014
% Delft University of Technology


    if ~exist('eta', 'var') || isempty(eta)
        eta = 0;
    end

    
    % Decode current solution
    ind = 1;
    model.pi  = reshape(x(ind:ind + numel(model.pi)  - 1), size(model.pi));  ind = ind + numel(model.pi);
    model.tau = reshape(x(ind:ind + numel(model.tau) - 1), size(model.tau)); ind = ind + numel(model.tau);    
    model.A   = reshape(x(ind:ind + numel(model.A)   - 1), size(model.A));   ind = ind + numel(model.A);
    model.W   = reshape(x(ind:ind + numel(model.W)   - 1), size(model.W));   ind = ind + numel(model.W);
    
    if  any(strcmpi(model.type, {'hidden_unit_logisitc_bgd_continuous', 'hidden_unit_logisitc_sgd_continuous'}))
        model.V      = reshape(x(ind:ind + numel(model.V)      - 1), size(model.V));   ind = ind + numel(model.V);
        model.b_bias    = reshape(x(ind:ind + numel(model.b_bias)    - 1), size(model.b_bias)); ind = ind + numel(model.b_bias);
        model.c_bias = reshape(x(ind:ind + numel(model.c_bias) - 1), size(model.c_bias));
    end
    
    is_InF_NaN(model.pi);
    is_InF_NaN(model.tau);
    is_InF_NaN(model.A);
    is_InF_NaN(model.W);
    is_InF_NaN(model.V);
    is_InF_NaN(model.b_bias);
    is_InF_NaN(model.c_bias);
    
    K = length(model.c_bias);
    D = size(model.W, 2);
    % disp(['K: ', num2str(K), '  D: ', num2str(D)]);
    
    grad_logZ = zeros(size(x));
    grad_logM = zeros(size(x));
    
    L = 0;
    
    % Loop over training sequences
    for i=1:length(train_X)
        
        % Perform forward-backward algorithm
        [alpha, beta, log_M, psi, log_h_mid_alpha, ~] = forward_backward_hidden_unit_logistic(train_X{i}, train_T(i), model, 0);
%             disp(['value of log_M: ', num2str(log_M)]);
        
        T = size(train_X{i}, 2);
        no_hidden = size(model.pi, 1);
        
        % Sum conditional log-likelihood    
        % clculate L - logZ
        % Compute log value of Z: sum_y {M}     
        log_M_temp = zeros(K, 1);    
        log_Xi_vec = cell(K, 1);
        for label=1:K
            [~, ~, log_M_temp(label, 1), ~, ~, log_Xi_vec{label, 1}] = forward_backward_hidden_unit_logistic(train_X{i}, label, model, 1);
        end
        max_v = max(log_M_temp);
        diff = log_M_temp - max_v;
        sum_diff = sum(exp(diff));
        log_Z = max_v + log(sum_diff);
        
        L = L + log_M - log_Z;
%         L = L - log_Z;

%         disp(['log_M: ', num2str(log_M), '  log_Z: ', num2str(log_Z), '  log_M-log_Z: ', num2str(log_M-log_Z)]);
%         disp(['L: ', num2str(L)]);
        
        
        
        % Compute gradients
        
        % to avoid Inf
        diff_logMtemp_logZ = log_M_temp - log_Z;


        for label = 1 : K
            
            y = zeros(K, 1);
            y(label, 1) = 1;
            
            Xi = log_Xi_vec{label,1};
            
            % gradient of pi, which is a H * 1 matrix
            grad_pi_logM_temp = exp(Xi(1, :, 2, 1)) + exp(Xi(1, :, 2, 2));
            is_InF_NaN(grad_pi_logM_temp);
            grad_pi_logM_temp = grad_pi_logM_temp';
%             grad_pi_logZ = bsxfun(@plus, grad_pi_logZ, exp(diff_logMtemp_logZ(label,1)) .* grad_pi_logM_temp);
            
            % gradient of tau, which is a H * 1 matrix
            grad_tau_logM_temp = exp(Xi(T-1, :, 1, 2)) + exp(Xi(T-1, :, 2, 2));
            grad_tau_logM_temp = grad_tau_logM_temp';
            is_InF_NaN(grad_tau_logM_temp);
            
            % gradient of model.A, which is a H * 1 matrix
            grad_A_logM_temp = sum(exp(Xi(1:(T-1), :, 2, 2)), 1);
            grad_A_logM_temp = grad_A_logM_temp';
            is_InF_NaN(grad_A_logM_temp);
            
            % gradient of model.W, which is a H * D matrix
            grad_W_logM_temp = zeros(no_hidden, D);
            for t = 1:T
                if t == 1
                    av = exp(Xi(T, :, 1, 2));
                    bv = (av') * (train_X{i}(:, t)');
                    grad_W_logM_temp = bsxfun(@plus, grad_W_logM_temp, bv);
                else
                    av = exp(Xi(t-1, :, 1, 2)) + exp(Xi(t-1, :, 2, 2));
                    bv = (av') * (train_X{i}(:, t)');
                    grad_W_logM_temp = bsxfun(@plus, grad_W_logM_temp, bv);
                end
            end
            is_InF_NaN(grad_W_logM_temp);
            
            % gradient of model.V, which is a H * K matrix
            grad_V_logM_temp = zeros(no_hidden, K);
            av = bsxfun(@plus, exp(Xi(:, :, 1, 2)), exp(Xi(:, :, 2, 2)));
            bv = sum(av, 1);
            grad_V_logM_temp = (bv') * (y'); % == -realmax?
            is_InF_NaN(grad_V_logM_temp);
            
            % gradient of model.c_bias, which is a  1 * K matrix
            grad_c_logM_temp = y';
            is_InF_NaN(grad_c_logM_temp);
                        
            % gradient of model.b_bias, which is a H * 1 matrix
            av = bsxfun(@plus, exp(Xi(:, :, 1, 2)), exp(Xi(:, :, 2, 2)));
            bv = sum(av, 1);
            grad_b_logM_temp = bv';
            
            grad_logM_temp = [grad_pi_logM_temp(:); grad_tau_logM_temp(:); grad_A_logM_temp(:); ...
                grad_W_logM_temp(:); grad_V_logM_temp(:); grad_b_logM_temp(:); grad_c_logM_temp(:)];
            is_InF_NaN(grad_logM_temp);
            
            grad_logZ = grad_logZ + exp(diff_logMtemp_logZ(label,1)) .* grad_logM_temp;
            is_InF_NaN(grad_logZ);
            
            if label == train_T(i)
                grad_logM = grad_logM + grad_logM_temp;
            end
            
        end
        
        if mod(i, 50) == 0
            disp([num2str(i), ' instances have been process!']);
        end
               
    end 

    
    % Return cost function and gradient
    C = -L + lambda .* sum(x .^ 2);
    dC = -(grad_logM - grad_logZ) + 2 .* lambda .* x;
%     for i=1:length(dC)
%         disp(['grad_logM: ', num2str(grad_logM(i,1)), ' grad_logZ: ', num2str(grad_logZ(i,1)), '  dC: ', num2str(dC(i,1))]);
%     end
    
    %update the parameters
    x = x - eta * dC;
    
%     disp(['-L: ', num2str(-L)]);
   
    

    
end


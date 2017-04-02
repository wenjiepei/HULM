function [ alpha, beta, log_M, psi, log_h_mid_alpha, log_Xi, mid_psi_result_rtn ] = forward_backward_hidden_unit_logistic_optimized( X, Label, model, if_calculate_xi, if_mid_psi, mid_psi_result )
%FORWARD_BACKWARD_HIDDEN_UNIT_LOGISTIC Performs forward-backward algorithm in an
%hidden_unit_logistic model
%
%   [alpha, beta, log_M, psi, log_h_mid_alpha, log_Xi] = forward_backward(X, model) 
%
% Performs the forward-backward algorithm on time series X in the
% hidden-unit logistic model.
% specified in model. The messages are returned in alpha and beta. The
% function also returns the normalization constants of the messages in rho,
% as well as the emission probabilities at all time steps.
%
%
% (C) Wenjie Pei, 2014
% Delft University of Technology

    % Initialize some variables
    T = size(X, 2); % the length of the chain
    no_hidden = numel(model.pi);
    K = size(model.V, 2);
    alpha = zeros(2, no_hidden, T); % alpha(1,:,:) corresponds to i==0, alpha(2,:,:) corresponds to i==1, where i is in the equation in the document
    beta  = zeros(2, no_hidden, T);
%     disp(['T: ', num2str(T), '  K: ', num2str(K)]);

    y = zeros(K, 1);
    y(Label, 1) = 1;
  
    % Compute log value of Psi
    if (if_mid_psi==0)
        psi = zeros(T, no_hidden, 2, 2); % T * H * 2 * 2 dimensions
        mid_psi_result_rtn = zeros(T, no_hidden, 2, 2);
        for t=1:T
            for i=1:2
                for j=1:2
                    [psi(t, :, i, j), mid_psi_result_rtn(t, :, i, j)] = calculate_psi_from_scratch(model, i-1, j-1, X, t, y);
                end
            end
        end
    else 
        mid_psi_result_rtn = 0;
        psi = zeros(T, no_hidden, 2, 2); % T * H * 2 * 2 dimensions
        for t=1:T
            for i=1:2
                for j=1:2
                    psi(t, :, i, j) = calculate_psi_using_mid_result(model, i-1, j-1, t, y, mid_psi_result(t, :, i, j));
                end
            end
        end
    end
    psi(1, :, 2, :) = psi(1, :, 1, :);
    is_InF_NaN(psi);
    
    % Compute message for first hidden variable
    % note:
    % the actual value: alpha(1, :, t) = exp(exp_coef1 + alpha(1, :, t))
    % the actual value: alpha(2, :, t) = exp(exp_coef2 + alpha(2, :, t))
    
    alpha_exp_coef = zeros(size(alpha));
    last_exp_coef1 = 0;
    last_exp_coef2 = max(model.W * X(:,1) + model.V * y + model.b_bias + model.pi, [], 1);
    last_exp_coef = max(last_exp_coef1, last_exp_coef2);
    alpha_exp_coef(1, :, 1) = last_exp_coef1;
    alpha_exp_coef(2, :, 1) = last_exp_coef2;
    
    % log value
    alpha(1, :, 1) = 0;
    tv = model.W * X(:,1) + model.V * y + model.b_bias + model.pi;
    alpha(2, :, 1) = (tv - last_exp_coef2)';
    
    
    % Perform forward pass
    for t=2:T
        exp_coef1 = last_exp_coef;
        pa1 = alpha(1, :, t-1) + last_exp_coef1 - exp_coef1;
        pa2 = alpha(2, :, t-1) + last_exp_coef2 - exp_coef1;
        alpha(1, :, t) = log( exp(pa1) + exp(pa2) );
        exp_coef2 = max( max(psi(t, :, 1, 2) + alpha(1, :, t-1)) + last_exp_coef1, max(psi(t, :, 2, 2) + alpha(2, :, t-1)) + last_exp_coef2   );
        pa1 = psi(t, :, 1, 2) - exp_coef2 + alpha(1, :, t-1) + last_exp_coef1;
        pa2 = psi(t, :, 2, 2) - exp_coef2 + alpha(2, :, t-1) + last_exp_coef2;
        alpha(2, :, t) = log(exp(pa1) + exp(pa2));
        last_exp_coef1 = exp_coef1;
        last_exp_coef2 = exp_coef2;
        last_exp_coef = max(last_exp_coef1, last_exp_coef2);
        alpha_exp_coef(1, :, t) = last_exp_coef1;
        alpha_exp_coef(2, :, t) = last_exp_coef2;
    end
    
    alpha(2, :, T) = alpha(2, :, T) + (model.tau)';
    is_InF_NaN(alpha);
    
    % Compute log value of M (sum_{z} exp(E))
    pa1 = alpha(1, :, T) + last_exp_coef1 - last_exp_coef;
    pa2 = alpha(2, :, T) + last_exp_coef2 - last_exp_coef;
    v_last = log( exp(pa1) + exp(pa2) );
    v_last = v_last';
    is_InF_NaN(v_last);
    log_h_mid_alpha = v_last + last_exp_coef;
    log_M = sum(log_h_mid_alpha) + model.c_bias * y;
    is_InF_NaN(log_M);
   
    
%     disp(['value of log_M: ', num2str(log_M)]);
% %     
%     % old code
%     alpha(1, :, 1) = 1;
%     alpha(2, :, 1) = exp(model.W * X(:,1) + model.V * y + model.b_bias + model.pi)'; 
%     
% 
%     % Perform forward pass
%     for t=2:T
%         alpha(1, :, t) = alpha(1, :, t-1) + alpha(2, :, t-1);
%         alpha(2, :, t) = exp(psi(t, :, 1, 2)) .* alpha(1, :, t-1) + exp(psi(t, :, 2, 2)) .* alpha(2, :, t-1);    
%     end
%     
%     alpha(2, :, T) = alpha(2, :, T) .* exp(model.tau)';
%     
%     % Compute log value of M (sum_{z} exp(E))
%     v_last = alpha(1, :, T) + alpha(2, :, T);
%     v_last = v_last';
%     log_h_mid_alpha = log(v_last);
%     log_M = sum(log_h_mid_alpha) + model.c_bias * y;
%     
% %     if abs(log_M) < 1.0e-10
% %         disp(['value of log_M: ', num2str(log_M)]);
% %         
% %     end
%     
%     disp(['value of log_M: ', num2str(log_M)]);
    
    beta_exp_coef = zeros(size(beta));
    last_exp_coef1 = 0;
    last_exp_coef2 = max(model.tau);
    beta_exp_coef(1, :, T) = last_exp_coef1;
    beta_exp_coef(2, :, T) = last_exp_coef2;
     
    % Perform backward pass
    % log value of beta
    beta(1, :, T) = 0;
    beta(2, :, T) = model.tau'-last_exp_coef2;
          
    for t=T-1:-1:1
        exp_coef1 = max(max(beta(1, :, t+1))+last_exp_coef1, max(psi(t+1, :, 1, 2)+beta(2, :, t+1) )+last_exp_coef2 );
        pa1 = beta(1, :, t+1) + last_exp_coef1 - exp_coef1;
        pa2 = beta(2, :, t+1) + last_exp_coef2 + psi(t+1, :, 1, 2) - exp_coef1;
        beta(1, :, t) = log(exp(pa1) + exp(pa2));
        
        exp_coef2 = max(max(beta(1, :, t+1))+last_exp_coef1, max(psi(t+1, :, 2, 2)+beta(2, :, t+1) )+last_exp_coef2 );
        pa1 = beta(1, :, t+1) + last_exp_coef1 - exp_coef2;
        pa2 = beta(2, :, t+1) + last_exp_coef2 + psi(t+1, :, 2, 2) - exp_coef2;
        beta(2, :, t) = log(exp(pa1) + exp(pa2));
        
        last_exp_coef1 = exp_coef1;
        last_exp_coef2 = exp_coef2;
        
        beta_exp_coef(1, :, t) = last_exp_coef1;
        beta_exp_coef(2, :, t) = last_exp_coef2;
    end
   
    [midv, ~] = calculate_psi_from_scratch(model, 0, 1, X, t, y);
    beta(2, :, 1) = beta(2, :, 1) + (model.pi)' + midv;
    is_InF_NaN(beta);
    
%     % for test: calculate log_M again with beta
%     last_exp_coef = max(last_exp_coef1, last_exp_coef2);
%     pa1 = beta(1, :, 1) + last_exp_coef1 - last_exp_coef;
%     pa2 = beta(2, :, 1) + last_exp_coef2 - last_exp_coef;
%     v_last = log(exp(pa1) + exp(pa2));
%     v_last = v_last'; 
%     log_M2 = sum(v_last+last_exp_coef) + model.c_bias * y;
%     disp(['value of log_M2: ', num2str(log_M2)]);
%     
%     % old codes
%     beta(1, :, T) = 1;
%     beta(2, :, T) = exp(model.tau)';
%     
%     for t=T-1:-1:1
%         beta(1, :, t) = beta(1, :, t+1) + exp(psi(t+1, :, 1, 2)) .* beta(2, :, t+1);
%         beta(2, :, t) = beta(1, :, t+1) + exp(psi(t+1, :, 2, 2)) .* beta(2, :, t+1);
%     end
%     
%     beta(2, :, 1) = beta(2, :, 1) .* exp(model.pi)' .* exp(calculate_psi(model, 0, 1, X, t, y))';
%     
%     % for test: calculate log_M again with beta
%     v_last = beta(1, :, 1) + beta(2, :, 1);
%     v_last = v_last';
%     log_M2 = sum(log(v_last)) + model.c_bias * y;
%     disp(['value of log_M2: ', num2str(log_M2)]);
    
    
    if if_calculate_xi==1
        % Calculate log value of \xi
        % note that: Xi(T, :, :, :) corrsponds to the Xi(0, :, :, :)
        log_Xi = zeros(T, no_hidden, 2, 2);
        temp = zeros(T, no_hidden, 2, 2);
        for t = 1:T
            for k = 1:2
                for j = 1:2
                    if t < T
                        temp(t, :, k, j) = bsxfun(@plus, alpha(k, :, t) + alpha_exp_coef(k, :, t), psi(t+1, :, k, j));
                        temp(t, :, k, j) = bsxfun(@plus, temp(t, :, k, j), beta(j, :, t+1) + beta_exp_coef(j, :, t+1) );
                    elseif t == T
                        temp(t, :, k, j) = beta(j, :, 1) + beta_exp_coef(j, :, 1);
                    end

                end
            end
        end
       

        for h =1:no_hidden
            v = sum(log_h_mid_alpha([1:h-1, h+1:end]));
            vmid = bsxfun(@plus, v, temp(:, h, :, :));
            sum_v = bsxfun(@plus, vmid, model.c_bias * y);
            log_Xi(:, h, :, :) = bsxfun(@minus, sum_v, log_M);
        end
        log_Xi(T, :, 2, :) = -realmax;
        is_InF_NaN(log_Xi);
        
    else 
        log_Xi = 0;
    end

end


function [v, mid_psi_result_rtn_h] = calculate_psi_from_scratch(model, z_t_1, z_t, X, n, y)

%     v = zeros(1, no_hidden);
    mid_psi_result_rtn_h = (z_t * z_t_1) .* model.A + z_t .* (model.W * X(:,n)) + z_t .* model.b_bias;
    mid_psi_result_rtn_h = mid_psi_result_rtn_h';
    if (z_t ~= 0)
        v = mid_psi_result_rtn_h + (model.V * y)';
    else
        v = mid_psi_result_rtn_h;
    end
 
end

function v = calculate_psi_using_mid_result(model, z_t_1, z_t, n, y, mid_psi_result_h)
    
    if (z_t ~= 0)
        v = mid_psi_result_h + (model.V * y)';
    else
        v = mid_psi_result_h;
    end
 
end
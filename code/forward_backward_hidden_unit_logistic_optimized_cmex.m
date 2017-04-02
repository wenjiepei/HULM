function [ alpha, beta, log_M, psi, log_h_mid_alpha, log_Xi, mid_psi_result_rtn ] = forward_backward_hidden_unit_logistic_optimized_cmex( X, Label, model, if_calculate_xi, if_mid_psi, mid_psi_result )
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
        mid_r = (model.V * y)';
        mid_psi_result_rtn = 0;
        psi(:, :, :, :) = calculate_psi(mid_r, mid_psi_result, T, no_hidden);
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
    
    [last_exp_coef, last_exp_coef1, last_exp_coef2, alpha, alpha_exp_coef] = calculate_alpha(alpha, alpha_exp_coef, psi, last_exp_coef1, last_exp_coef2, T, no_hidden);
    
    alpha(2, :, T) = alpha(2, :, T) + (model.tau)';
    is_InF_NaN(alpha);
    
    % Compute log value of M (sum_{z} exp(E))
    pa1 = alpha(1, :, T) + last_exp_coef1 - last_exp_coef;
    pa2 = alpha(2, :, T) + last_exp_coef2 - last_exp_coef;
    max_pa = max(pa1, pa2);
    v_last = max_pa + log( exp(pa1-max_pa) + exp(pa2-max_pa) );
%     v_last = log( exp(pa1) + exp(pa2) );
    v_last = v_last';
    is_InF_NaN(v_last);
    log_h_mid_alpha = v_last + last_exp_coef;
    log_M = sum(log_h_mid_alpha) + model.c_bias * y;
    is_InF_NaN(log_M);
    
    beta_exp_coef = zeros(size(beta));
    last_exp_coef1 = 0;
    last_exp_coef2 = max(model.tau);
    beta_exp_coef(1, :, T) = last_exp_coef1;
    beta_exp_coef(2, :, T) = last_exp_coef2;
     
    % Perform backward pass
    % log value of beta
    beta(1, :, T) = 0;
    beta(2, :, T) = model.tau'-last_exp_coef2;
    
    [beta, beta_exp_coef] = calculate_beta(beta, beta_exp_coef, psi, last_exp_coef1, last_exp_coef2, T, no_hidden);
   
    [midv, ~] = calculate_psi_from_scratch(model, 0, 1, X, 1, y);
    beta(2, :, 1) = beta(2, :, 1) + (model.pi)' + midv;
    is_InF_NaN(beta);
    
    if if_calculate_xi==1
        % Calculate log value of \xi
        % note that: Xi(T, :, :, :) corrsponds to the Xi(0, :, :, :)
        log_Xi = zeros(T, no_hidden, 2, 2);
        vmid = mid_calculate_xi(alpha, alpha_exp_coef, beta, beta_exp_coef, psi, log_h_mid_alpha, T, no_hidden);
        sum_v = bsxfun(@plus, vmid, model.c_bias * y);
        log_Xi(:, :, :, :) = bsxfun(@minus, sum_v, log_M);
        
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

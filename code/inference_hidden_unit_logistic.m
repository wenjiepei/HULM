function [ pred_label, probability ] = inference_hidden_unit_logistic( X, model )
%INFERENCE_HIDDEN_UNIT_LOGISTIC predicts the label based on input model.
%   [predicted_label] = inference_hidden_unit_logistic(X, model)


    K = length(model.c_bias);
    D = size(model.W, 2);
    T = size(X, 2);
    no_hidden = size(model.pi, 1);
    
    log_M_temp = zeros(K, 1);
 
    for label=1:K
        [~, ~, log_M_temp(label, 1), ~, ~, ~] = forward_backward_hidden_unit_logistic(X, label, model, 0);
    end
    [max_v, index] = max(log_M_temp);
    diff = log_M_temp - max_v;
    sum_diff = sum(exp(diff));
    log_Z = max_v + log(sum_diff);
%     disp(['log_Z: ', num2str(log_Z)]);
    
    L = max_v - log_Z;
    
%     prob_all = exp(log_M_temp - log_Z)
%     
%     sumv = sum(prob_all)
    
    probability = exp(L);
    pred_label = index;
%    disp(['Predicted Label: ', num2str(pred_label), ' with probability: ' num2str(probability)]);


end


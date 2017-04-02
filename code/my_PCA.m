function [ pca_X ] = my_PCA( X, top_num )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

[coef,~,latent] = princomp(X, 'econ');
if ~exist('top_num', 'var') || isempty(top_num)
    top_num = size(coef, 2);
end
cumsum(latent(1:top_num))./sum(latent)
segment_coef = coef(:, 1:top_num);

pca_X = X * segment_coef;

end


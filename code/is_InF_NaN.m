function [ yes_no ] = is_InF_NaN( input_matrix )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if numel(find(isinf(input_matrix)) > 0)
    yes_no = 1;
    error('InF!');
elseif numel(find(isnan(input_matrix)) > 0)
    yes_no = 2;
    error('NaN!');
else
    yes_no = 0;


end


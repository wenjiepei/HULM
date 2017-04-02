function [ output_args ] = compute_average_time_series_length( dataset_name )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

X=cell(0);
switch dataset_name
    case 'CK'
        load(['data' filesep 'CK1' filesep 'ck_shape.mat']);
    case 'Action'
        load(['data' filesep 'Action3D' filesep 'joint_feat_coordinate.mat']);
        X = feat;
    case 'Activity'
        load(['data' filesep 'Activity' filesep 'joint3D_feature_noFFT.mat']);
        X = Joint3D_feature;
    otherwise
        error('do not support this dataset.');
end
    
total_frame = 0;
for i=1:length(X)
    total_frame = total_frame + size(X{i}, 2);
end

avg_frame = total_frame / length(X)

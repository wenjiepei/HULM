
function visualization_banana_results(no_hidden, mode)
clc;
addpath('/Users/wenjie/Documents/matlab_toolboxes/export_fig');
if ~exist('mode', 'var') || isempty(mode)
    mode = 'train';
end
if ~exist('no_hidden', 'var') || isempty(no_hidden)
    no_hidden = 5;
end
dataset_name = 'banana';
sample_size = 2000;
lambda = 0;
load([ '..' filesep 'data' filesep dataset_name filesep, 'banana_' num2str(sample_size) '.mat']);
load(['result' filesep dataset_name '_' num2str(sample_size) filesep num2str(no_hidden) filesep 'classification_results_lambda_' num2str(lambda) '.mat']);
if exist(['result' filesep dataset_name '_' num2str(sample_size) filesep num2str(no_hidden) filesep 'boundary_points.mat'])
    load(['result' filesep dataset_name '_' num2str(sample_size) filesep num2str(no_hidden) filesep 'boundary_points.mat']);
else
    boundary = get_boundary(dataset_name, no_hidden, sample_size, lambda);
end
results = [];
if strcmp(mode, 'test')
   data = zeros(length(test_X), 2);
   for i = 1:length(test_X)
       data(i, :) = test_X{i}(:, 1);
   end
   results = test_results;
   T = test_T;
else
    data = zeros(length(train_X), 2);
    for i = 1:length(train_X)
        data(i, :) = train_X{i}(:, 1);
    end
    results = train_results;
    T = train_T;
end
max_x = max(data(:, 1));
max_y = max(data(:, 2));
min_x = min(data(:, 1));
min_y = min(data(:, 2));
bb = 1.1;
max_x = bb*max_x;
min_x = bb*min_x;
max_y = bb*max_y;
min_y = bb*min_y;
[~, label_1] = find(T==1);
[~, label_2] = find(T==2);
% r = find(results(:, 1) == results(:, 2));
% w = find(results(:, 1) ~= results(:, 2));
% r_1 = intersect(r, label_1);
% r_2 = intersect(r, label_2);
% w_1 = intersect(w, label_1);
% w_2 = intersect(w, label_2);
% ss = 5;
% figure(no_hidden);
% scatter(data(r_1, 1), data(r_1, 2), ss, 'r^');
% hold on;
% scatter(data(r_2, 1), data(r_2, 2), ss, 'b*');
% hold on;
% scatter(data(w_1, 1), data(w_1, 2), ss, 'b^');
% hold on;
% scatter(data(w_2, 1), data(w_2, 2), ss, 'r*');
figure(no_hidden);
ss = 5;
scatter(data(label_1, 1), data(label_1, 2), ss, 'r^');
hold on;
scatter(data(label_2, 1), data(label_2, 2), ss, 'b+');
hold on;
for i = 1:length(boundary)/2
   plot(boundary(2*i-1:2*i, 1), boundary(2*i-1:2*i, 2), 'k-', 'LineWidth', 2);
   hold on;
end
axis([min_x, max_x, min_y, max_y]);
set(gcf,'color','white');
axis off;
grid off;
box off;

save_path = ['result' filesep dataset_name '_' num2str(sample_size) filesep num2str(no_hidden) filesep 'HULM_banana_nStates_' num2str(no_hidden) '_sample_' num2str(sample_size) '.pdf'];
export_fig(save_path, '-pdf');

end

function boundary = get_boundary(dataset_name, no_hidden, sample_size, lambda)
%%% load the model
index_fold = 1;
load(['result' filesep dataset_name '_', num2str(sample_size) filesep num2str(no_hidden) filesep 'sgd_model_temp_cross_validation_',...
    num2str(lambda) '_' num2str(index_fold), '.mat']);

%%% get the value range for features
load([ '..' filesep 'data' filesep dataset_name filesep, 'banana_' num2str(sample_size) '.mat']);
max_x = train_X{1}(1,1);
min_x = train_X{1}(1,1);
max_y = train_X{1}(2,1);
min_y = train_X{1}(2,1);
for i = 2:length(train_X)
    max_x = max(train_X{i}(1, 1), max_x);
    min_x = min(train_X{i}(1, 1), min_x);
    max_y = max(train_X{i}(2, 1), max_y);
    min_y = min(train_X{i}(2, 1), min_y);
end
max_x = 1.1*max_x;
min_x = 1.1*min_x;
max_y = 1.1*max_y;
min_y = 1.1*min_y;

%%% uniformaly sample the points in the area
sample_width = 200;
unit_x = (max_x - min_x) / (sample_width-1);
unit_y = (max_y - min_y) / (sample_width-1);
points_mat = [];
points = cell(sample_width);
time_s = clock;
for i = 1:sample_width
    point(1, 1:2) = min_x + (i-1)*unit_x;
    for j = 1:sample_width
        point(2, 1:2) = min_y + (j-1)*unit_y;
        points{i, j} = point;
%         points_mat(i,j) = point;
    end
end

%%% classify the sample points
pred = zeros(sample_width, sample_width);
for i = 1:sample_width
    for j = 1:sample_width
        pred(i, j) = inference_hidden_unit_logistic_optimized(points{i, j}, model);
    end
    if mod(i, 10)==0
        time_e = clock;
        disp([num2str(i) ' finished! with time: ', num2str(etime(time_e, time_s))]);
    end
end

% [I1, J1] = find(pred==1);
% [I2, J2] = find(pred>1);
% ss = 5;
% figure(55);
% for i = 1:length(I1)
%     scatter(points{I1(i), J1(i)}(1, 1), points{I1(i), J1(i)}(2, 1), ss, 'r^');
%     hold on;
% end
% for i = 1:length(I2)
%     scatter(points{I2(i), J2(i)}(1, 1), points{I2(i), J2(i)}(2, 1), ss, 'b^');
%     hold on;
% end

%%% get the boundaries
boundary = [];
for i = 1:sample_width-1
    for j = 1:sample_width-1
        if pred(i,j) ~= pred(i+1, j)
            point = (points{i, j}(:, 1) + points{i+1, j}(:, 1)) / 2;
            pointb = point' + [0, -unit_y/2];
            pointe = point' + [0, unit_y/2];
            boundary = [boundary; pointb];
            boundary = [boundary; pointe];
        end
        if pred(i, j)~=pred(i, j+1)
            point = (points{i, j}(:, 1) + points{i, j+1}(:, 1)) / 2;
            pointb = point' + [-unit_x/2, 0];
            pointe = point' + [unit_x/2, 0];
            boundary = [boundary; pointb];
            boundary = [boundary; pointe];
        end
    end
end
figure(55);
for i = 1:length(boundary)/2
   plot(boundary(2*i-1:2*i, 1), boundary(2*i-1:2*i, 2));
   hold on;
end

save(['result' filesep dataset_name '_' num2str(sample_size) filesep num2str(no_hidden) filesep 'boundary_points.mat'], 'points', 'pred', 'boundary');


end








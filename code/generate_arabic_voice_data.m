%%% generate the arabic_voice_data from arabic data
function generate_arabic_voice_data(if_permutate)
clc;

load(['..' filesep 'data' filesep 'arabic' filesep 'training_set_arabic.mat']);
load(['..' filesep 'data' filesep 'arabic' filesep 'test_set_arabic.mat']);
training_set = normalize_data(training_set);
test_set = normalize_data(test_set);

new_data_ind = [];
new_X = [];
new_labels = [];
for s = 1:66
    % for train data
    for d = 1:10
        st = (d-1)*660+(s-1)*10 + 1;
        se = st + 10 - 1;
        new_X = [new_X training_set(st:se)];
        new_labels = [new_labels zeros(1, 10)+s];
    end
end

for s = 1:22
    % for test data
    for d = 1:10
        st = (d-1)*220+(s-1)*10 + 1;
        se = st + 10 - 1;
        new_X = [new_X test_set(st:se)];
        new_labels = [new_labels zeros(1, 10)+s+66];
    end
end

% window feature
combined_value = 3;
for i=1:length(new_X)
    new_X{1, i} = get_overlap_feature(new_X{1, i}, combined_value);
end

%%% randomly permutate the samples for each subjects
if if_permutate
    for s = 1:88
        randp = randperm(100);
        start = (s-1)*100 + 1;
        temp = new_X(start:start+99);
        new_X(start:start+99) = temp(randp);
        temp = new_labels(start:start+99);
        new_labels(start:start+99) = temp(randp);
    end
end


save(['..' filesep 'data' filesep 'arabic_voice' filesep 'arabic_voice_window_3_ifperm_' num2str(if_permutate) '.mat'], 'new_X', 'new_labels');
end


% method: standardization: x-mean(x) / standard_deviation
function normalized_X = normalize_data(X)
% convert X into a big matrix whole_X
whole_X = X{1};
for i = 2:length(X)
    whole_X = [whole_X, X{i}];
end
max_v = max(whole_X, [], 2)

mean_feature = mean(whole_X, 2)
std_feature = std(whole_X, 1, 2)
for i = 1:length(X)
    X{i} = X{i} - repmat(mean_feature, 1, size(X{i}, 2));
    X{i} = X{i} ./ repmat(std_feature, 1, size(X{i}, 2));
end
normalized_X = X;

% check the max value
whole_X = X{1};
for i = 2:length(X)
    whole_X = [whole_X, X{i}];
end
max_v = max(whole_X, [], 2)
end

function [overlap_feature] = get_overlap_feature(feat, times)

T = size(feat, 2);
D = size(feat, 1);
overlap_feature = zeros(D*times, T-times+1);
for i=1:size(overlap_feature, 2)
    for d = 1:times
        overlap_feature(D*(d-1)+1:D*d, i) = [feat(:, i+d-1)];
    end
    
end

end
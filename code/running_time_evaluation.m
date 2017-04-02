%%% test the running time w.r.t. the hidden state number 
% to check whether it follows the square relationship

function running_time_evaluation()
clc;
clear all;

hidden_states = [1, 8, 16, 32, 64, 128, 256];
seed = 100;
s = RandStream.create('mt19937ar','seed',seed);
RandStream.setGlobalStream(s);
dataset_name = 'CK';
times = zeros(size(hidden_states));
for h = 1:length(hidden_states)
    if strcmp(dataset_name, 'arabic')
        times(h) = running_arabic_experiment(hidden_states(h));
    elseif strcmp(dataset_name, 'character')
        times(h) = running_character_experiment(hidden_states(h));
    elseif strcmp(dataset_name, 'CK')
        times(h) = running_CK_experiment(hidden_states(h));
    end
end
disp('the running time is: ');
for i = 1:length(times)
    disp([num2str(hidden_states(i)), '  ', num2str(times(i))]);
end

%%% visualize the results
figure(1);
plot(hidden_states, times);

%%% save results
file_dir = ['result' filesep 'running_time_evaluation' filesep dataset_name];
if ~exist(file_dir)
    mkdir(file_dir);
end
save([file_dir filesep 'HULM_running_time.mat'], 'times');
end

% test with arabic dataset
function total_time = running_arabic_experiment(nStates)
dataset_name = 'arabic';
disp(['Begin the experiments  for arabic with nState: ', num2str(nStates)]);
lambda = 0;

load(['data' filesep 'arabic' filesep 'training_set_arabic']);
load(['data' filesep 'arabic' filesep 'test_set_arabic']);
% combine feature
for i=1:length(training_set)
    training_set{1, i} = get_overlap_feature(training_set{1, i}, combined_value);
end
for i=1:length(test_set)
    test_set{1, i} = get_overlap_feature(test_set{1, i}, combined_value);
end

training_size = 500;
perm = randperm(length(train_labels));
% perm = randperm(1000);
perm = perm(1:training_size);
train_X = training_set(perm);
train_T = train_labels(perm);
test_X  = test_set;
test_T  = test_labels;
disp(['feature dimension: ' num2str(size(train_X{1}, 1)), '   classes: ', num2str(max(train_T))]);

annealing = 0.5;
eta = 1e-3;
batch_size = 5;
max_iter = 1;
index_fold = 1;
if_lambda = 0;
stop_threshold = 5;
decay_threshold = 15;
dataset_name = 'arabic';
time_s = clock;
model = train_hidden_unit_logisitic_sgd2(train_X, train_T, test_X, test_T, ...
    lambda, max_iter, eta, batch_size, nStates, index_fold, if_lambda, dataset_name, annealing, decay_threshold, stop_threshold, false);
time_e = clock;
total_time = etime(time_e, time_s);
disp(['the running time with ' num2str(nStates) ' is: ' num2str(total_time)]);

end

% test with character dataset
function total_time = running_character_experiment(nStates)
dataset_name = 'arabic';
disp(['Begin the experiments for character with nState: ', num2str(nStates)]);
lambda = 0;

load(['..' filesep 'data' filesep 'character' filesep 'randomperm_character_overlap.mat']);
disp(['size of dataset: ', num2str(length(new_X))]);

randp = randperm(length(new_X));
randp = randp(1:100);
train_X = new_X(randp);
train_T = new_labels(randp);
test_X  = new_X(randp);
test_T  = new_labels(randp);
disp(['feature dimension: ' num2str(size(train_X{1}, 1)), '   classes: ', num2str(max(train_T))]);
disp(['trainning set for final cross-validation : ' num2str(length(train_X)), '    test set: ' num2str(length(test_X))]);

annealing = 0.5;
eta = 1e-3;
batch_size = 5;
max_iter = 1;
index_fold = 1;
if_lambda = 0;
stop_threshold = 5;
decay_threshold = 15;
dataset_name = 'character';
time_s = clock;
model = train_hidden_unit_logisitic_sgd2(train_X, train_T, test_X, test_T, ...
    lambda, max_iter, eta, batch_size, nStates, index_fold, if_lambda, dataset_name, annealing, decay_threshold, stop_threshold, false);
time_e = clock;
total_time = etime(time_e, time_s);
disp(['the running time with ' num2str(nStates) ' is: ' num2str(total_time)]);

end

% test with character dataset
function total_time = running_CK_experiment(nStates)
dataset_name = 'CK';
disp(['Begin the experiments for CK with nState: ', num2str(nStates)]);
lambda = 0;

load(['data' filesep 'CK' filesep 'randomperm_CK.mat']);
disp(['size of dataset: ', num2str(length(new_X))]);

randp = randperm(length(new_X));
% randp = randp(1:100);
train_X = new_X(randp);
train_T = new_labels(randp);
test_X  = new_X(randp);
test_T  = new_labels(randp);
disp(['feature dimension: ' num2str(size(train_X{1}, 1)), '   classes: ', num2str(max(train_T))]);
disp(['trainning set for final cross-validation : ' num2str(length(train_X)), '    test set: ' num2str(length(test_X))]);

annealing = 0.5;
eta = 1e-3;
batch_size = 5;
max_iter = 1;
index_fold = 1;
if_lambda = 0;
stop_threshold = 5;
decay_threshold = 15;
dataset_name = 'character';
time_s = clock;
model = train_hidden_unit_logisitic_sgd2(train_X, train_T, test_X, test_T, ...
    lambda, max_iter, eta, batch_size, nStates, index_fold, if_lambda, dataset_name, annealing, decay_threshold, stop_threshold, false);
time_e = clock;
total_time = etime(time_e, time_s);
disp(['the running time with ' num2str(nStates) ' is: ' num2str(total_time)]);

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


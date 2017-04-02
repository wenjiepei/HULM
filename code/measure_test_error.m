function test_err = measure_test_error(test_X, test_T, model)

disp('measuring test error...');
t1 = clock;
tot = 0; test_err = 0;
tot = tot + length(test_T);
for i=1:length(test_X)
    test_err = test_err + (inference_hidden_unit_logistic_optimized(test_X{i}, model) ~= test_T(i));
end
test_err = test_err / tot;
disp(['Classification error (test set): ' num2str(test_err)]);
t2 = clock;
disp(['classification time: ', num2str(etime(t2, t1))]);

end
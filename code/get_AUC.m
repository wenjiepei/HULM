
function auc = get_AUC(true_labels, posclass, test_score)
[X,Y,T,auc] = perfcurve(true_labels, test_score, posclass);
end
function EER = my_EER(FRR, FAR)
%%%%% calculation of EER value
if FRR <= 1.0
    FRR = 100 * FRR;
end
if FAR <= 1.0
    FAR = 100 * FAR;
end
tmp1=find (FRR-FAR<=0);
tmps=length(tmp1);

if ((FAR(tmps)-FRR(tmps))<=(FRR(tmps+1)-FAR(tmps+1)))
    EER=(FAR(tmps)+FRR(tmps))/2;tmpEER=tmps;
else
    EER=(FRR(tmps+1)+FAR(tmps+1))/2;tmpEER=tmps+1;
end
end
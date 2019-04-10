function B = testKNNH(X, KNNHparam, test_sample)

if (isfield(KNNHparam, 'pcaW') == 1)
    V = X*KNNHparam.pcaW;
else
    V = X;
end

if test_sample == 1
    U = V*KNNHparam.r;
else
    U = KNNHparam.B;
end
B = (U>0);

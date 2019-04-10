%%
clc
clear
rng('shuffle')
addpath('./KNNH/', './tools/');

%% SETTING
num_bits = 16;
dataset = 'mnist'; % cifar10, mnist, labelme, places
feature_type = 'gist'; % gist, vggfc7, uint (MINIST only), alexnet (Places205 only)

gpuDevice_ID = 1; % if CPU-only / you don't want to use GPU, just set it as -1

ITER = 10; % run ITER times
K = 20; % 200 for Places205
%%

use_gpu = check_gpu(gpuDevice_ID);

dataset_name = [dataset,'-', feature_type];

fprintf('KNNH with %d bits | %s | K : %d.\n', num_bits, upper(dataset_name), K);

[train_features, train_labels, query_features, query_labels] = load_dataset( dataset_name );

v_Pr2 = zeros(ITER,1);
v_mAP = zeros(ITER,1);
v_P1K = zeros(ITER,1);
mAP_cls = 0;

for iter = 1:ITER
    clear KNNHparam
    
    % RUN EMBEDDING
    % zero-mean
    avg = mean(train_features, 2);
    X = bsxfun(@minus, train_features, avg);
    Q = bsxfun(@minus, query_features, avg);

    % Preprocess data to remove correlated features.
    Xcov = X*X';
    Xcov = (Xcov + Xcov')/(2*size(X, 2));
    [U,S,~] = svd(Xcov);
    
    if (strcmp(dataset, 'places'))
        KNNHparam.idX = load_idX('./model/KNN_Places205.mat', use_gpu, K, X');
    end

    X_pca = U(:,1:num_bits)' * X; % d'*n
    Q_pca = U(:,1:num_bits)' * Q;

    KNNHparam.nbits = num_bits;
    KNNHparam.K = K;
    KNNHparam.times = 1;
    KNNHparam.gpu = use_gpu;
    KNNHparam = trainKNNH(X_pca', KNNHparam);
    gallery_code = testKNNH(X_pca', KNNHparam, 0);
    test_code    = testKNNH(Q_pca', KNNHparam, 1);
         
    [Pr2, Ptop1000, mAP, mAP_cls] = evaluation(gallery_code, train_labels, test_code, query_labels, mAP_cls);
             
    fprintf('%d bits: r=2: %f - top1k: %f - mAP: %f \n', num_bits, Pr2, Ptop1000, mAP);
    v_Pr2(iter) = Pr2;
    v_mAP(iter) = mAP;
    v_P1K(iter) = Ptop1000;
end
fprintf('  MEAN : r=2: %f - top1k: %f - mAP: %f \n', num_bits, mean(v_Pr2), mean(v_P1K), mean(v_mAP));
fprintf('  STD  : r=2: %f - top1k: %f - mAP: %f \n', num_bits, std(v_Pr2)*100, std(v_P1K)*100, std(v_mAP)*100);
% mAP_cls./ITER

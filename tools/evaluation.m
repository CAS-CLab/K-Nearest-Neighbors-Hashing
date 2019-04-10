function [Pr2, Ptop1000, mAP, mAP_cls] = evaluation(gallery_code, gallery_label, test_code, test_label, mAP_cls)

num_test_sample  = length(test_label);
precisionat2     = zeros(num_test_sample, 1);
precisiontop1000 = zeros(num_test_sample, 1);
mAPs = zeros(num_test_sample, 1);

parfor i=1:num_test_sample
    dist = sum(bsxfun(@xor,gallery_code,test_code(i, :)), 2);
    [sorted_dist,idx] = sort(dist);
    sorted_gallery_label = gallery_label(idx);
    true_label = test_label(i);
    
    relavant_retrieval = (sorted_gallery_label == true_label);
    
    % For precision at r
    num_retrieval_img = sum(sorted_dist <= 2.0);
    if (num_retrieval_img > 0) 
        precisionat2(i) = sum(relavant_retrieval(1:num_retrieval_img))/num_retrieval_img;
    else
        precisionat2(i) = 0;
    end
    % For precision at topk return
    precisiontop1000(i) = sum(relavant_retrieval(1:1000))/1000;
    
    % For AvgP
    cum_precision = cumsum(relavant_retrieval)./(1:length(gallery_label))';
    mAPs(i) = sum(cum_precision.*relavant_retrieval)/sum(relavant_retrieval);
end

num_class = max(test_label);
Pr_class       = zeros(num_class, 1);
Ptop1000_class = zeros(num_class, 1);
mAP_class      = zeros(num_class, 1);

for c=0:num_class
    num_class_samples     = sum(int32(test_label == c));
    Pr_class(c + 1)       = sum(precisionat2(test_label == c))/num_class_samples;
    Ptop1000_class(c + 1) = sum(precisiontop1000(test_label == c))/num_class_samples;
    mAP_class(c + 1)      = sum(mAPs(test_label == c))/num_class_samples;
end

Pr2      = mean(Pr_class);
Ptop1000 = mean(Ptop1000_class);
mAP      = mean(mAP_class);
mAP_cls  = mAP_cls + mAP_class;

end

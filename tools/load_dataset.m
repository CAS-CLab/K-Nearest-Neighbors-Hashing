function [ train_features, train_labels, test_features, test_labels] = load_dataset(dataset)

switch dataset
    case 'cifar10-gist'
        load(['./datasets/cifar10_gist512_train.mat']);
        load(['./datasets/cifar10_gist512_test.mat']);
    case 'cifar10-vggfc7'
        load('./datasets/cifar10_vggfc7_train.mat');
        load('./datasets/cifar10_vggfc7_test.mat');
    case 'mnist-gist'
        load('./datasets/mnist_gist512_train.mat');
        load('./datasets/mnist_gist512_test.mat');
    case 'mnist-uint'
        load('./datasets/mnist_intensity_train.mat');
        load('./datasets/mnist_intensity_test.mat');
    case 'labelme-gist'
        load('./datasets/labelme_gist512_train.mat');
        load('./datasets/labelme_gist512_test.mat');
    case 'labelme-vggfc7'
        load('./datasets/labelme_vggfc7_train.mat');
        load('./datasets/labelme_vggfc7_test.mat');
    case 'places-alexnet'
        load('./datasets/places205-train.mat');
        load('./datasets/places205-test.mat');
        load('./datasets/places205-train-label.mat');
        load('./datasets/places205-test-label.mat');
        test_labels    = TEST_LABEL_MAT;
        train_labels   = TRAIN_LABEL_MAT;
        train_features = TRAIN_GIST_MAT';
        test_features  = TEST_GIST_MAT';
    otherwise
        error('Undefined dataset')
end
train_features = train_features';
test_features = test_features';

end

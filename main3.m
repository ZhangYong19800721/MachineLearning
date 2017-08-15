clear all;
close all;

[train_images,~,test_images,test_labels] = learn.import_mnist('./+learn/mnist.mat');
K = 10; [D,S,M] = size(train_images);
train_labels = eye(10); train_labels = repmat(train_labels,1,10,M);

configure.stacked_rbm = [D,500,500];
configure.softmax_rbm = [K,500,2000];
dbn = learn.DBN(configure);

parameters.learn_rate = [1e-8 1e-2];
parameters.weight_cost = 1e-4;
parameters.max_it = 1e0;
dbn = dbn.pretrain(train_images,train_labels,parameters);

% save('dbn_pretrain.mat','dbn');
load('dbn_pretrain.mat');

y = dbn.classify(test_images);
error1 = sum(y~=test_labels') / length(y);

parameters.max_it = 1e6;
dbn = dbn.train(train_images,train_labels,parameters);
save('dbn_train.mat','dbn');
y = dbn.classify(test_images);
error2 = sum(y~=test_labels') / length(y);
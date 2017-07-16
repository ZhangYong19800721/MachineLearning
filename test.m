clear all;
close all;

[train_images,train_labels,test_images,test_labels] = import_mnist('mnist.mat');

load('dbn.mat');

y = dbn.classify(test_images);

error = sum(y~=test_labels') / length(y);
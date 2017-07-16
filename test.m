clear all;
close all;

[train_images,train_labels,test_images,test_labels] = import_mnist('mnist.mat');

load('sae.mat');

y = sae.rebuild(test_images);

r_error = sum(sum(abs(y - test_images))) / size(test_images,2);
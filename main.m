clear all;
close all;

[train_images,train_labels,test_images,test_labels] = learn.import_mnist('+learn/mnist.mat');
points = reshape(train_images,784,[]);        %points = points(:,1:100);
labels = double(reshape(train_labels,1,[]));  %labels = labels(:,1:100);

configure = [784,1000,1];
perception = learn.Perception(configure);
perception = perception.initialize();

lmbp = learn.ConjugateGradientBP(points,labels,perception);

weight = optimal.ConjugateGradient(lmbp,lmbp,perception.weight,1e-5,1e2,1e-6,1e5);
perception.weight = weight;
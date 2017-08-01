clear all;
close all;

[train_images,train_labels,test_images,test_labels] = learn.import_mnist('+learn/mnist.mat');
points = reshape(train_images,784,[]);
labels = double(reshape(train_labels,1,[]));

configure = [784,500,500,2000,1];
perception = learn.Perception(configure);
perception = perception.initialize();

lmbp = learn.ConjugateGradientBP(points,labels,perception);

weight = optimal.ConjugateGradient(lmbp,lmbp,perception.weight,1e-5,1e-6,1e5);
perception.weight = weight;

figure(3);
y = perception.do(x);
plot(x,l,'b'); hold on;
plot(x,y,'r.'); hold off;

e = norm(l - y,2);
clear all;
close all;

[train_images] = import_mnist('mnist.mat');
[N,S,M] = size(train_images);
for n = 1:M
    mnist{n} = train_images(:,:,n);
end

dbn = ML.DeepBeliefNet([784,500,500]);
dbn = dbn.pretrain(mnist,1e-6,0.1,1e6);



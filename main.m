clear all;
close all;

[train_images,train_labels,test_images,test_labels] = import_mnist('mnist.mat');
[D,S,M] = size(train_images);

for m = 1:M
    L = zeros(10,100);
    I = sub2ind(size(L),1+train_labels(:,m),[1:100]');
    L(I) = 1;
    mnist{m} = [L;train_images(:,:,m)];
end

configure.stacked_rbm = [784,500,500];
configure.softmax_rbm = [10,500,2000];
dbn = ML.DeepBeliefNet(configure);
dbn = dbn.pretrain(mnist,1e-4,0.1,1e6);

save('dbn.mat','dbn');



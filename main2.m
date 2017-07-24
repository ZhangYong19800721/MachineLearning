% 2017-02-15
% author£ºZhangYong, 24452861@qq.com

clear all;
close all;

load('sae_pre.mat');

[train_images,train_labels,test_images,test_labels] = import_mnist('mnist.mat');
[D,S,M] = size(train_images);

for m = 1:M
%     L = zeros(10,100);
%     I = sub2ind(size(L),1+train_labels(:,m),[1:100]');
%     L(I) = 1;
%     mnist{m} = [L;train_images(:,:,m)];
    mnist{m} = train_images(:,:,m);
end

sae = sae.train(mnist,1e-4,0.005,1e6);

save('sae.mat','sae');

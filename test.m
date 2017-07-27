clear all;
close all;

[train_images,train_labels,test_images,test_labels] = import_mnist('mnist.mat');

train_images = reshape(train_images,784,542*100);
train_labels = reshape(train_labels,1,542*100);

% for n = 100:200
%     image = reshape(uint8(255 * train_images(:,n)),28,28)';
%     imshow(image);
%     train_labels(n)
% end

load('sae_pre.mat');

train_code = sae.encode(train_images);
test_code = sae.encode(test_images);

for n = 1:1e4
    code = test_code(:,n);
%     diff = repmat(code,1,54200) - train_code;
%     dist = sqrt(sum(diff.^2));
    diff = xor(repmat(code,1,54200),train_code);
    dist = sum(diff);
    [min_dis,min_idx] = min(dist);
    y(n) = train_labels(min_idx);
end

error = sum(y~= test_labels')/1e4;
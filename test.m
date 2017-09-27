clear all
close all

load('images.mat'); % points = points(1:(32*32),:);
load('sae_pretrained.mat');

[D,N] = size(points);
points = double(points) / 255;
code = sae.encode(points,'fix');

query_id = 4000;
query_code = code(:,query_id);
query_code = repmat(query_code,1,N);
match_idx = sum(xor(query_code,code)) <= 0;

dir = 'D:\imagebasev3\';
seq = 1:N;
match = seq(match_idx);
k = 0;
for n = match
    k = k + 1
    file = strcat(strcat(dir,sprintf('%09d',n-1)),'.jpg');
    image = imread(file);
    imshow(imresize(image,0.5));
end
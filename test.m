clear all
close all

load('images.mat');
load('sae_pretrain.mat');

[D,N] = size(points);
points = double(points) / 255;
code = sae.encode(points,'fix');

query_id = 3;
query_code = code(:,query_id);
query_code = repmat(query_code,1,N);
match_idx = sum(xor(query_code,code)) <= 5;

dir = 'D:\��Ƶ����\02-TrainData\imagebasev3\';
seq = 1:N;
match = seq(match_idx);
for n = match
    file = strcat(strcat(dir,sprintf('%09d',n-1)),'.jpg');
    image = imread(file);
    imshow(image);
end
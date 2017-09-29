clear all;
close all;
rng(1);

load('images.mat'); points = double(points)/255; N = size(points,2);
load('code.mat');

query_id = 15;
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
    imshow(imresize(image,0.2));
end
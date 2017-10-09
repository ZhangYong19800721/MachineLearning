clear all;
close all;
rng(1);

load('images.mat'); points = double(points)/255; N = size(points,2);
load('ps.mat');
load('thresh.mat');

code = ps.compute(points,3);
code = code > repmat(thresh,1,size(code,2));
query_id = 2002;
query_code = code(:,query_id);
query_code = repmat(query_code,1,N);
match_idx = sum(xor(query_code,code)) <= 0;

dir = 'D:\imagebasev3\';
seq = 1:N;
match = seq(match_idx);
k = 0;
for n = match
    k = k + 1
    fileQ = strcat(strcat(dir,sprintf('%09d',query_id-1)),'.jpg');
    fileK = strcat(strcat(dir,sprintf('%09d',n-1))       ,'.jpg');
    imageQ = imread(fileQ);
    imageK = imread(fileK);
    subplot(2,1,1);
    imshow(imresize(imageQ,0.2));
    subplot(2,1,2);
    imshow(imresize(imageK,0.2));
end
clear all
close all

load('tiny_images.mat'); 
[D,S,M] = size(points);
points = double(points)/255;
load('sae_trained.mat')

points = reshape(points,D,[]);
code = sae.encode(points,'nosample');
thresh = zeros(64,1);

for n = 1:64
    c = learn.cluster.KMeansPlusPlus(code(n,:),2);
    thresh(n) = mean(c);
end

code = code > repmat(thresh,1,size(code,2));
save('thresh.mat','thresh')
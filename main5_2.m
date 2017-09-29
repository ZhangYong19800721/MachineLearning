clear all
close all

load('images.mat'); points = double(points)/255;
load('sae_trained.mat')

code = sae.encode(points,'nosample');
t = zeros(64,1);

for n = 1:64
    c = learn.cluster.KMeansPlusPlus(code(n,:),2);
    t(n) = mean(c);
end

code = code > repmat(t,1,size(code,2));
save('code.mat','code')
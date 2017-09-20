clear all
close all
rng(1)

load('tiny_images.mat')
D = 32*32*3;  % 数据维度
S = 1000;  % minibatch的大小
M = 2000;  % minibatch的个数
N = S*M;   % 图片总数
points = zeros(D,S,M,'uint8');

for m = 1:M
    disp(sprintf('minibatch_id = %d',m));
    images = learn.data.loadTinyImages((m-1)*S+[1:S],'D:\迅雷下载\tiny_images.bin'); % 每次载入S幅图像
    for i = 1:S
        yuv = rgb2ycbcr(images(:,:,:,i));
        points(:,i,m) = yuv(:);
    end
end

save('tiny_images.mat','points','-v7.3');
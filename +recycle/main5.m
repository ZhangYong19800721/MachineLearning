clear all
close all
rng(1)

load('tiny_images.mat')
D = 32*32*3;  % ����ά��
S = 1000;  % minibatch�Ĵ�С
M = 2000;  % minibatch�ĸ���
N = S*M;   % ͼƬ����
points = zeros(D,S,M,'uint8');

for m = 1:M
    disp(sprintf('minibatch_id = %d',m));
    images = learn.data.loadTinyImages((m-1)*S+[1:S],'D:\Ѹ������\tiny_images.bin'); % ÿ������S��ͼ��
    for i = 1:S
        yuv = rgb2ycbcr(images(:,:,:,i));
        points(:,i,m) = yuv(:);
    end
end

save('tiny_images.mat','points','-v7.3');
clear all
close all

dir = 'D:\imagebasev3\视频基因训练数据\imagebasev3\';
N = 73605;
points = uint8(zeros(32*32*3,N));
parfor n = 1:N
    disp(sprintf('image %d',n));
    file = strcat(strcat(dir,sprintf('%09d',n-1)),'.jpg');
    image = imread(file);
    image = rgb2ycbcr(image);
    image = imresize(image,[32,32]);
    points(:,n) = image(:);
end

save('images.mat','points');
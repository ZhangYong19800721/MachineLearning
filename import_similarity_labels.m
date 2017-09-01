%% 读入相似性数据
s0 = csvread('similarity_part00.txt');
s1 = csvread('similarity_part01.txt');
s2 = csvread('similarity_part02.txt');
s3 = csvread('similarity_part03.txt');
s4 = csvread('similarity_part04.txt');
s5 = csvread('similarity_part05.txt');
s6 = csvread('similarity_part06.txt');
s7 = csvread('similarity_part07.txt');
s8 = csvread('similarity_part08.txt');

labels = [s0; s1; s2; s3; s4; s5; s6; s7; s8]';

% dir = 'D:\imagebasev3\视频基因训练数据\imagebasev3\';
% for n = 1:size(labels,2)
%     x = labels(1,n); y = labels(2,n); 
%     file1 = strcat(strcat(dir,sprintf('%09d',x)),'.jpg');
%     file2 = strcat(strcat(dir,sprintf('%09d',y)),'.jpg');
%     image1 = imread(file1);
%     image2 = imread(file2);
%     subplot(2,1,1); imshow(image1);
%     subplot(2,1,2); imshow(image2);
% end

labels(1,:) = labels(1,:) + 1;
labels(2,:) = labels(2,:) + 1;

clear s1 s2 s3 s4 s5 s6 s7 s8;

%% 计算传递的相似性
[~,N] = size(labels);
ssets = cell(1,N);
for n = 1:N
    ssets{n} = labels(1:2,n);
end

for i = 1:N
    disp(num2str(i));
    if isempty(ssets{i})
        % do nothing
    else
        for j = (i+1):N
            if isempty(intersect(ssets{i},ssets{j}))
                % do nothing
            else
                ssets{i} = union(ssets{i},ssets{j});
                ssets{j} = [];
            end
        end
    end
end

save;

labels = [];
for n = 1:N
    disp(num2str(n));
    if isempty(ssets{n})
        % do nothing
    else
        M = numel(ssets{n});
        for i = 1:M
            for j = (i+1):M
                labels = [labels [ssets{n}(i) ssets{n}(j) 1]'];
            end
        end
    end
end

save('labels_pos.mat',labels);


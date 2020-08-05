function [center_points, labels]=KMeansPP(points,K,opt)
%KMEANSPLUSPLUS KMeans++聚类算法，KMeans的升级版本
%   参考文献“k-means++: The Advantages of Careful Seeding”

% K,分类个数
% points,数据样本点（训练数据）
% center_points, 经过训练后得到的中心点
% points_labels, 所有的训练数据加标签后的结果
    %% 参数设置
    disp('kmeans++ ...');
    if nargin <= 2 % 没有给出参数
        opt = [];
        disp('使用默认参数集');
    end
    
    if ~isfield(nargin,'epsilon') % 给出参数但是没有给出epsilon
        opt.epsilon = 1e-5; 
        disp(sprintf('使用默认epsilon值%f',opt.epsilon));
    end
    
    if ~isfield(nargin,'maxit') % 给出参数但是没有给出maxit
        opt.maxit = 1e+6; 
        disp(sprintf('使用默认maxit值%d',opt.maxit));
    end

    %% 使用k-means++的方法初始化随机起始点
    [~, N]=size(points);  %D是数据维数，N是样本点个数
    center_points_idx = randi(N); %第1个中心点从所有的数据中随机取一个
    center_points = points(:,center_points_idx); % 初始化第1个中心点
    disp(sprintf('select the %05d-th center',1));
    min_distance = inf(1,N);
    while length(center_points_idx) < K
        c = length(center_points_idx);
        disp(sprintf('select the %05d-th center',c+1));
        distance = sum((points - center_points(:,c)).^2,1);
        min_distance = min([min_distance; distance]);
        if sum(min_distance) == 0
            prob = ones(size(min_distance)) ./ N;
        else
            prob = min_distance ./ sum(min_distance);
        end
        value = rand();
        for n = 1:N 
            value = value - prob(n);
            if value < 0
                break;
            end
        end
        center_points_idx = [center_points_idx n]; 
        center_points = [center_points points(:,n)];
    end
    clearvars center_points_idx c center_point_c prob value;
    
    %% 使用k-means进行聚类
    distance = inf(K, N);
    for it = 1:opt.maxit
        for k = 1:K
		    % disp(sprintf('compute distance for %05d-th center',k));
            distance(k,:) = sqrt(sum((points - center_points(:,k)).^2,1));
        end
        
        [min_distance, labels] = min(distance);
        old_center_points = center_points;
        
        if mod(it,100) == 0
            save(sprintf('center_points_%08d.mat', it), 'center_points');
        end
        
        for k = 1:K
            center_points(:,k) = sum(points(:,labels == k),2) / sum(labels == k);
        end
        
        all_shift = sqrt(sum((old_center_points - center_points).^2, 1));
        max_shift = max(all_shift);
        obj_value = sum(min_distance);
        disp(sprintf('iteration: %08d, shift:%16.8f, obj_fun:%16.8f', it, max_shift, obj_value));
        if max_shift <= opt.epsilon
            break;
        end
    end
end


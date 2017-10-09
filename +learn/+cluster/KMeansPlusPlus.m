function [center_points, labels]=KMeansPlusPlus(points,K)
%KMEANSPLUSPLUS KMeans++聚类算法，KMeans的升级版本
%   参考文献“k-means++: The Advantages of Careful Seeding”

% K,分类个数
% points,数据样本点（训练数据）
% center_points, 经过训练后得到的中心点
% points_labels, 所有的训练数据加标签后的结果
    disp('kmeans++ ...');

    [~, N]=size(points);  %D是数据维数，N是样本点个数

    % 使用k-means++的方法初始化随机起始点
    center_points_idx = randi(N); %第1个中心点从所有的数据中随机取一个
    center_points = points(:,center_points_idx); % 初始化第1个中心点
    distance_matrix = inf(K,N);
    while length(center_points_idx) < K
        c = length(center_points_idx);
        disp(sprintf('c = %d',c));
        center_point_c = repmat(center_points(:,c),1,N);
        delta = points - center_point_c;
        distance_matrix(c,:) = sum(delta.^2,1);
     
        min_distance_matrix = min(distance_matrix);
        if sum(min_distance_matrix) == 0
            prob = ones(size(min_distance_matrix)) ./ N;
        else
            prob = min_distance_matrix ./ sum(min_distance_matrix);
        end
        cprob = [0 cumsum(prob)];
        idx = sum(cprob < rand());
        while sum(center_points_idx==idx) > 0 % 表示在center_points_init_idx集合中能找到idx
            idx = sum(cprob < rand());             % 那就重新抽样一次直到idx是一个新值
        end
        center_points_idx = [center_points_idx idx]; 
        center_points = [center_points points(:,idx)];
    end
    
    distance_matrix = zeros(K,N);
    
    while true
        parfor k = 1:K
            center_point_k = repmat(center_points(:,k),1,N);
            delta = points - center_point_k;
            distance_matrix(k,:) = sqrt(sum(delta.^2,1));
        end
        
        [~,min_idx] = min(distance_matrix);
        
        old_center_points = center_points;
        parfor k = 1:K
            if(sum(min_idx == k) == 0)
                center_points(:,k) = old_center_points(:,k);
            else
                center_points(:,k) = sum(points(:,min_idx == k),2) / sum(min_idx == k);
            end
        end
        
        norm_value = norm(old_center_points - center_points);
        disp(sprintf('norm-value:%f',norm_value));
        if norm_value <= 0.0001
            parfor k = 1:K
                center_point_k = repmat(center_points(:,k),1,N);
                delta = points - center_point_k;
                distance_matrix(k,:) = sqrt(sum(delta.^2));
            end
            
            [~,labels] = min(distance_matrix);
            break;
        end
    end
end


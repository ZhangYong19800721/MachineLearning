function [center_points, labels]=KMeansPlusPlus(points,K)
%KMEANSPLUSPLUS KMeans++�����㷨��KMeans�������汾
%   �ο����ס�k-means++: The Advantages of Careful Seeding��

% K,�������
% points,���������㣨ѵ�����ݣ�
% center_points, ����ѵ����õ������ĵ�
% points_labels, ���е�ѵ�����ݼӱ�ǩ��Ľ��
    disp('kmeans++ ...');

    [~, N]=size(points);  %D������ά����N�����������

    % ʹ��k-means++�ķ�����ʼ�������ʼ��
    center_points_idx = randi(N); %��1�����ĵ�����е����������ȡһ��
    center_points = points(:,center_points_idx); % ��ʼ����1�����ĵ�
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
        while sum(center_points_idx==idx) > 0 % ��ʾ��center_points_init_idx���������ҵ�idx
            idx = sum(cprob < rand());             % �Ǿ����³���һ��ֱ��idx��һ����ֵ
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


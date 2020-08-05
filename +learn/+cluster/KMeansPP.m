function [center_points, labels]=KMeansPP(points,K,opt)
%KMEANSPLUSPLUS KMeans++�����㷨��KMeans�������汾
%   �ο����ס�k-means++: The Advantages of Careful Seeding��

% K,�������
% points,���������㣨ѵ�����ݣ�
% center_points, ����ѵ����õ������ĵ�
% points_labels, ���е�ѵ�����ݼӱ�ǩ��Ľ��
    %% ��������
    disp('kmeans++ ...');
    if nargin <= 2 % û�и�������
        opt = [];
        disp('ʹ��Ĭ�ϲ�����');
    end
    
    if ~isfield(nargin,'epsilon') % ������������û�и���epsilon
        opt.epsilon = 1e-5; 
        disp(sprintf('ʹ��Ĭ��epsilonֵ%f',opt.epsilon));
    end
    
    if ~isfield(nargin,'maxit') % ������������û�и���maxit
        opt.maxit = 1e+6; 
        disp(sprintf('ʹ��Ĭ��maxitֵ%d',opt.maxit));
    end

    %% ʹ��k-means++�ķ�����ʼ�������ʼ��
    [~, N]=size(points);  %D������ά����N�����������
    center_points_idx = randi(N); %��1�����ĵ�����е����������ȡһ��
    center_points = points(:,center_points_idx); % ��ʼ����1�����ĵ�
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
    
    %% ʹ��k-means���о���
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


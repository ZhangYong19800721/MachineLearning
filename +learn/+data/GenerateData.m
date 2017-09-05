classdef GenerateData
    methods(Static)
        function [points,labels] = type0(N)
            %
            points = 10 * rand(2,N); % 产生x->[0,10],y->[0,10]上均匀分布的随机数
            labels = 2 * (sqrt(points(1,:).^2 + points(2,:).^2) <= 5) - 1;
        end
        
        function [points,labels] = type1(N)
            %GENERATEDATA 产生训练数据
            % 非线性可分双弧形
            points1 = [];
            while true
                p(1) = 20 * rand() - 10; p(2) = 10 * rand();
                r = sqrt(p(1).^2+p(2).^2);
                if 8 <= r && r <= 10
                    points1 = [points1 p'];
                end
                
                [~,K1] = size(points1);
                if K1 >= N/2
                    break;
                end
            end
            points1(1,:) = points1(1,:) - 3;
            points1(2,:) = points1(2,:) - 2;
            
            points2 = [];
            while true
                p(1) = 20 * rand() - 10; p(2) = -10 * rand();
                r = sqrt(p(1).^2+p(2).^2);
                if 8 <= r && r <= 10
                    points2 = [points2 p'];
                end
                
                [~,K2] = size(points2);
                if K2 >= N/2
                    break;
                end
            end
            points2(1,:) = points2(1,:) + 3;
            points2(2,:) = points2(2,:) + 2;
            
            points = [points1 points2];
            labels = [ones(1,N/2) -ones(1,N/2)];
        end
        
        function [points,labels] = type2(N)
            % 
            points = 20 * rand(2,N) - repmat([10 10]',1,N); % 产生x->[-10,10],y->[-10,10]上均匀分布的随机数
            labels = [];
            for i = 1:N
                for j = (i+1):N
                    if points(2,i) < -5 && points(2,j) < -5
                        labels = [labels [i,j,+1]'];
                    elseif -5 <= points(2,i) && points(2,i) < 0 && -5 <= points(2,j) && points(2,j) < 0
                        labels = [labels [i,j,+1]'];
                    elseif  0 <= points(2,i) && points(2,i) < 5 &&  0 <= points(2,j) && points(2,j) < 5
                        labels = [labels [i,j,+1]'];
                    elseif  5 <= points(2,i) && points(2,i) <10 &&  5 <= points(2,j) && points(2,j) <10
                        labels = [labels [i,j,+1]'];
                    else
                        labels = [labels [i,j,-1]'];
                    end
                end
            end
        end
        
        function [points,labels] = type3(N)
            %GENERATEDATA 产生训练数据
            % 非线性可分双弧形
            points1 = [];
            while true
                p(1) = 20 * rand() - 10; p(2) = 10 * rand();
                r = sqrt(p(1).^2+p(2).^2);
                if 8 <= r && r <= 10
                    points1 = [points1 p'];
                end
                
                [~,K1] = size(points1);
                if K1 >= N/2
                    break;
                end
            end
            points1(1,:) = points1(1,:) + 3;
            points1(2,:) = points1(2,:) - 2;
            
            points2 = [];
            while true
                p(1) = 20 * rand() - 10; p(2) = -10 * rand();
                r = sqrt(p(1).^2+p(2).^2);
                if 8 <= r && r <= 10
                    points2 = [points2 p'];
                end
                
                [~,K2] = size(points2);
                if K2 >= N/2
                    break;
                end
            end
            points2(1,:) = points2(1,:) - 3;
            points2(2,:) = points2(2,:) + 2;
            
            points = [points1 points2];
            labels = [-ones(1,N/2) ones(1,N/2)];
        end
        
        function [points,labels] = type4(N)
            % 一饼形
            points = 20 * rand(2,N) - repmat([10 10]',1,N);
            labels = -ones(1,N);
            for i = 1:N
                if norm(points(:,i),2) < 5
                    labels(i) = 1;
                end
            end
        end
        
        function [points,labels] = type5(N)
            % 
            K = N/4;
            points1 = rand(2,K); points1(1,:) = 20 * points1(1,:) - 10; points1(2,:) = 5 * points1(2,:) - 10;
            points2 = rand(2,K); points2(1,:) = 20 * points2(1,:) - 10; points2(2,:) = 5 * points2(2,:) -  5;
            points3 = rand(2,K); points3(1,:) = 20 * points3(1,:) - 10; points3(2,:) = 5 * points3(2,:) -  0;
            points4 = rand(2,K); points4(1,:) = 20 * points4(1,:) - 10; points4(2,:) = 5 * points4(2,:) +  5;
            
            points = [points1 points2 points3 points4];
            
            plot(points1(1,:),points1(2,:),'+'); hold on;
            plot(points2(1,:),points2(2,:),'.');
            plot(points3(1,:),points3(2,:),'*');
            plot(points4(1,:),points4(2,:),'o'); 
            
            labels = [];
            for i = 1:N
                for j = (i+1):N
                    if points(2,i) < -5 && points(2,j) < -5
                        labels = [labels [i,j,+1]'];
                    elseif -5 <= points(2,i) && points(2,i) < 0 && -5 <= points(2,j) && points(2,j) < 0
                        labels = [labels [i,j,+1]'];
                    elseif  0 <= points(2,i) && points(2,i) < 5 &&  0 <= points(2,j) && points(2,j) < 5
                        labels = [labels [i,j,+1]'];
                    elseif  5 <= points(2,i) && points(2,i) <10 &&  5 <= points(2,j) && points(2,j) <10
                        labels = [labels [i,j,+1]'];
                    else
                        labels = [labels [i,j,-1]'];
                    end
                end
            end
        end
        
        function [points,labels] = type6(N)
            % 有相近y值的两个点标记为相似
            points = 20 * rand(2,N) - repmat([10 10]',1,N); % 产生x->[-10,10],y->[-10,10]上均匀分布的随机数
            labels = [];
            for i = 1:N
                for j = (i+1):N
                    if abs(points(2,i) - points(2,j)) < 5
                        labels = [labels [i,j,+1]'];
                    else
                        labels = [labels [i,j,-1]'];
                    end
                end
            end
        end
        
        function [points,labels] = type7(N)
            % 有相近2范数的两个点标记为相似（到原点的距离相近）
            points = 20 * rand(2,N) - repmat([10 10]',1,N); % 产生x->[-10,10],y->[-10,10]上均匀分布的随机数
            labels = [];
            for i = 1:N
                for j = (i+1):N
                    if abs(norm(points(2,i),2) - norm(points(2,j),2)) < 3
                        labels = [labels [i,j,+1]'];
                    else
                        labels = [labels [i,j,-1]'];
                    end
                end
            end
        end
        
        function [points,labels] = type8(N)
            % 有相近2范数的两个点标记为相似（到原点的距离相近）
            K = N / 4; 
            r1 = 2.5 * rand(1,K) + 0;   a1 = 2*pi*rand(1,K); points1 = [r1;r1] .* [cos(a1);sin(a1)];
            r2 = 2.5 * rand(1,K) + 2.5; a2 = 2*pi*rand(1,K); points2 = [r2;r2] .* [cos(a2);sin(a2)];
            r3 = 2.5 * rand(1,K) + 5;   a3 = 2*pi*rand(1,K); points3 = [r3;r3] .* [cos(a3);sin(a3)];
            r4 = 2.5 * rand(1,K) + 7.5; a4 = 2*pi*rand(1,K); points4 = [r4;r4] .* [cos(a4);sin(a4)];
            
            plot(points1(1,:),points1(2,:),'+'); hold on;
            plot(points2(1,:),points2(2,:),'*');
            plot(points3(1,:),points3(2,:),'.');
            plot(points4(1,:),points4(2,:),'o'); 
            
            points = [points1 points2 points3 points4];
            labels = [];
            for i = 1:N
                for j = (i+1):N
                    norm_pi = norm(points(:,i),2); 
                    norm_pj = norm(points(:,j),2); 
                    if norm_pi <= 2.5 && norm_pj <= 2.5
                        labels = [labels [i,j,+1]'];
                    elseif 2.5 < norm_pi && norm_pi <= 5 && 2.5 < norm_pj && norm_pj <= 5
                        labels = [labels [i,j,+1]'];
                    elseif 5 < norm_pi && norm_pi <= 7.5 && 5 < norm_pj && norm_pj <= 7.5
                        labels = [labels [i,j,+1]'];
                    elseif 7.5 < norm_pi && 7.5 < norm_pj
                        labels = [labels [i,j,+1]'];
                    else
                        labels = [labels [i,j,-1]'];
                    end
                end
            end
        end
        
        function [points,labels] = type9(N)
            % 一饼形
            points = 20 * rand(2,N) - repmat([10 10]',1,N);
            labels = [];
            for i = 1:N
                for j = (i+1):N
                    if norm(points(:,i),2) < 5 && norm(points(:,j),2) < 5
                        labels = [labels [i,j,+1]'];
                    elseif 5 <= norm(points(:,i),2) && 5 <= norm(points(:,j),2)
                        labels = [labels [i,j,+1]'];
                    else
                        labels = [labels [i,j,-1]'];
                    end
                end
            end
        end
    end
end

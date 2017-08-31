classdef GenerateData
    methods(Static)
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
    end
end

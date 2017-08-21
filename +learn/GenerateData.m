classdef GenerateData
    methods(Static)
        function [points,labels] = type1(N)
            %GENERATEDATA 产生训练数据
            %
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
            points = 20 * rand(2,N) - repmat([10 10]',1,N);
            labels = ones(1,N);
            for i = 1:N
                if abs(points(1,i) - points(2,i)) > 2
                    labels(i) = -1;
                end
            end
        end
    end
end

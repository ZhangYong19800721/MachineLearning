classdef Perception
    %PERCEPTION 感知器
    %   
    
    properties
        weight;      % 一维数组，所有层的权值和偏置值都包含在这个一维数组中
        num_hidden;  % num_hidden{m}表示第m层的隐神经元个数
        num_visual;  % num_visual{m}表示第m层的显神经元个数
        star_w_idx;  % star_w_idx{m}表示第m层的权值的起始位置
        stop_w_idx;  % stop_w_idx{m}表示第m层的权值的结束位置
        star_b_idx;  % star_b_idx{m}表示第m层的偏置的起始位置
        stop_b_idx;  % stop_b_idx{m}表示第m层的偏置的结束位置
    end
    
    methods % 构造函数
        function obj = Perception(configure)
            % Perception 构造函数
            M = length(configure) - 1;
            obj.num_hidden{1} = configure(2);
            obj.num_visual{1} = configure(1);
            obj.star_w_idx{1} = 1;
            obj.stop_w_idx{1} = obj.num_hidden{1} * obj.num_visual{1};
            obj.star_b_idx{1} = obj.stop_w_idx{1} + 1;
            obj.stop_b_idx{1} = obj.stop_w_idx{1} + obj.num_hidden{1};
            
            for m = 2:M
                obj.num_hidden{m} = configure(m+1);
                obj.num_visual{m} = configure(m+0);
                obj.star_w_idx{m} = obj.stop_b_idx{m-1} + 1;
                obj.stop_w_idx{m} = obj.stop_b_idx{m-1} + obj.num_hidden{m} * obj.num_visual{m};
                obj.star_b_idx{m} = obj.stop_w_idx{m} + 1;
                obj.stop_b_idx{m} = obj.stop_w_idx{m} + obj.num_hidden{m};
            end
        end
    end
    
    methods
        function obj = initialize(obj)
            M = length(obj.num_hidden); % 得到层数
            for m = 1:M
                obj.weight(obj.star_w_idx{m}:obj.stop_w_idx{m}) = 0.01 * randn(size(obj.star_w_idx{m}:obj.stop_w_idx{m})); % 将权值初始化为0附近的随机数
                obj.weight(obj.star_b_idx{m}:obj.stop_b_idx{m}) = zeros(size(obj.star_b_idx{m}:obj.stop_b_idx{m})); % 将偏置值初始化为0
            end
        end
        
        function [y,a] = do(obj,x)
            % 多层感知器的计算过程
            % y 是最后的输出
            
            M = length(obj.num_hidden); % 得到层数
            a = cell(1,M);          
            for m = 1:M
                w = reshape(obj.weight(obj.star_w_idx{m}:obj.stop_w_idx{m}),obj.num_hidden{m},obj.num_visual{m});
                b = reshape(obj.weight(obj.star_b_idx{m}:obj.stop_b_idx{m}),obj.num_hidden{m},1);
                n = w * x + repmat(b,1,size(x,2));
                if m < M
                    a{m} = learn.sigmoid(n);
                    x = a{m};
                else
                    a{m} = n;
                    y = n;
                end
            end
        end
    end
    
    methods(Static)
        function [perception,e] = unit_test1()
            clear all;
            close all;
            N = 2000;
            x = linspace(-2,2,N);
            k = 6;
            f = @(x)sin(k * pi * x / 4);
            l = f(x);
            
            configure = [1,6,1];
            perception = learn.Perception(configure);
            perception = perception.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,perception.do(x)); hold off;
            
            lmbp = learn.LevenbergMarquardtBP(x,l,perception);
            
            weight = optimal.LevenbergMarquardt(lmbp,lmbp,perception.weight,1e-2,2e3);
            perception.weight = weight;
            
            figure(3);
            y = perception.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            e = norm(l - y,2);
        end
        
        function [perception,e] = unit_test2()
            clear all;
            close all;
            N = 1000;
            r1 = 2 * rand(1,N) + 8; a1 =  pi * rand(1,N); group1 = repmat(r1,2,1) .* [cos(a1); sin(a1)];
            r2 = 2 * rand(1,N) + 8; a2 = -pi * rand(1,N); group2 = repmat(r2,2,1) .* [cos(a2); sin(a2)];
            group1(1,:) = group1(1,:) + 4; group1(2,:) = group1(2,:) - 2;   
            group2(1,:) = group2(1,:) - 4; group2(2,:) = group2(2,:) + 2; 
            figure(1);
            plot(group1(1,:),group1(2,:),'+'); hold on;
            plot(group2(1,:),group2(2,:),'o'); hold off;
            
            configure = [2,20,1];
            perception = learn.Perception(configure);
            perception = perception.initialize();
            
            points = [group1 group2];
            labels = [ones(1,N) zeros(1,N)];
            
            lmbp = learn.LevenbergMarquardtBP(points,labels,perception);
            perception.weight = optimal.LevenbergMarquardt(lmbp,lmbp,perception.weight,1e-2,2e2);
            
            y = perception.do(points) > 0.5;
            e = sum(xor(y,labels)) / length(y);
        end
    end
end
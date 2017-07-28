classdef SingleLayerPerception
    %SINGLELAYERPERCEPTION 单层感知器
    %   
    
    properties
        weight; % 网络的权值(最后一列为偏置值)
        bias;   % 偏置值
    end
    
    methods % 构造函数
        function obj = SingleLayerPerception(input_num,output_num)
            obj.weight = 0.001 * randn(output_num,input_num);
            obj.bias = zeros(output_num,1);
        end
    end
    
    methods
        function obj = train(obj,minibatchs,learn_rate_min,learn_rate_max,max_it)
            minibatch_num = length(minibatchs);
            ob_window_size = minibatch_num;
            ob = ML.Observer('均方误差',1,ob_window_size,'xxx');
            
            learn_rate = learn_rate_max; % 初始化学习速度为最大学习速度
            error_list = zeros(1,ob_window_size);
            for minibatch_idx = 1:minibatch_num  % 初始化重建误差列表的移动平均值
                minibatch = minibatchs{minibatch_idx};
                error = norm(minibatch.labels - obj.do(minibatch.points),2);
                error_list(minibatch_idx) = error;
            end
            error_ave_old = mean(error_list);
            ob = ob.initialize(error_ave_old);
            
            for it = 0:max_it
                minibatch_idx = mod(it,minibatch_num) + 1;
                labels = minibatchs{minibatch_idx}.labels;
                points = minibatchs{minibatch_idx}.points; 
                [~,minibatch_size] = size(points);
                y = obj.do(points);
                error = labels - y; norm_e = norm(error,2);
                error_list(minibatch_idx) = norm_e;
                error_ave_new = mean(error_list);
                
                if minibatch_idx == minibatch_num
                    if error_ave_new > error_ave_old
                        learn_rate = 0.5 * learn_rate;
                        if learn_rate < learn_rate_min
                            break;
                        end
                    end
                    error_ave_old = error_ave_new;
                end
                
                description = strcat(strcat(strcat('迭代次数:',num2str(it)),'/'),num2str(max_it));
                description = strcat(description,strcat('学习速度:',num2str(learn_rate)));
                ob = ob.showit(error_ave_new,description);
                
                delta = error .* y .* (1 - y);
                delta_w = 2 * delta * points' / minibatch_size;
                delta_b = 2 * delta * ones(minibatch_size,1) / minibatch_size;
                obj.weight = obj.weight + learn_rate * delta_w;
                obj.bias   = obj.bias   + learn_rate * delta_b;
            end
        end
        
        function y = do(obj,x)
            [~,N] = size(x);
            b = repmat(obj.bias,1,N);
            y = ML.sigmoid(obj.weight * x + b);
        end
    end
    
    methods(Static)
        function [slp,e] = unit_test()
            clear all;
            close all;
            points = 100 * rand(2,1e4);
            labels = points(1,:) > points(2,:);
            
            figure(1);
            group1 = points(:, labels);
            group2 = points(:,~labels);
            plot(group1(1,:),group1(2,:),'x'); hold on;
            plot(group2(1,:),group2(2,:),'o'); hold off;
            
            for n = 1:100
                minibatchs{n}.labels = labels(:,(100*(n-1)+1):(100*(n-1)+100));
                minibatchs{n}.points = points(:,(100*(n-1)+1):(100*(n-1)+100));
            end
            slp = ML.SingleLayerPerception(2,1);
            slp = slp.train(minibatchs,1e-4,1e-1,1e4);
            
            y = slp.do(points) > 0.5;
            e = sum(xor(labels,y)) / length(y);
        end
    end
end


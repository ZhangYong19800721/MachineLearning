classdef Perception
    %PERCEPTION 感知器
    %   
    
    properties
        stacked_rbm; 
    end
    
    methods % 构造函数
        function obj = Perception(configure)
            obj.stacked_rbm = ML.StackedRestrictedBoltzmannMachine(configure);
        end
    end
    
    methods
        function obj = initialize(obj)
            layer_num = obj.stacked_rbm.layer_num(); % 得到层数
            for layer_idx = 1:layer_num
                num_hidden = obj.stacked_rbm.rbm_layers{layer_idx}.num_hidden;
                num_visual = obj.stacked_rbm.rbm_layers{layer_idx}.num_visual;
                obj.stacked_rbm.rbm_layers{layer_idx}.weight = 0.01 * randn(num_hidden,num_visual);
                obj.stacked_rbm.rbm_layers{layer_idx}.hidden_bias = zeros(num_hidden,1);
            end
        end
        
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
                
                delta = obj.BP(points,labels);
%                 delta_w = 2 * delta * points' / minibatch_size;
%                 delta_b = 2 * delta * ones(minibatch_size,1) / minibatch_size;
%                 obj.weight = obj.weight + learn_rate * delta_w;
%                 obj.bias   = obj.bias   + learn_rate * delta_b;
            end
        end
        
        function y = do(obj,x)
            y = obj.stacked_rbm.posterior(x);
        end
    end
    
    methods (Access = private)
        function delta = BP(obj,points,labels)
            % 反向传播算法
            layer_num = obj.stacked_rbm.layer_num(); % 得到层数
            N = size(points,2); % 共有N个数据样本
            
            % 首先执行正向计算,并记录每一层的输出
            n{1} = points; a{1} = points;
            for layer_idx = 1:layer_num
                n{layer_idx+1} = obj.stacked_rbm.rbm_layers{layer_idx}.foreward(a{layer_idx});
                a{layer_idx+1} = ML.sigmoid(n{layer_idx+1});
            end
            
            % 计算顶层的敏感性
            
            
        end
    end
    
    methods(Static)
        function [perception,e] = unit_test()
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
            
            perception = ML.Perception([2,100,1]);
            perception = perception.initialize();
            perception = perception.train(minibatchs,1e-4,1e-1,1e4);
            
            y = perception.do(points) > 0.5;
            e = sum(xor(labels,y)) / length(y);
        end
    end
end


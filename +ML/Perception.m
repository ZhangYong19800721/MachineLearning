classdef Perception
    %PERCEPTION 感知器
    %   
    
    properties
        weight; % 元胞数组，weight{m}表示第m层的权值
        bias;   % 元胞数组，bias{m}表示第m层的偏置值
    end
    
    methods % 构造函数
        function obj = Perception(configure)
            % Perception 构造函数
            M = length(configure) - 1;
            for m = 1:M
                obj.weight{m} = zeros(configure(m+1),configure(m));
                obj.bias{m}   = zeros(configure(m+1),1);
            end
        end
    end
    
    methods
        function obj = initialize(obj)
            M = length(obj.bias); % 得到层数
            for m = 1:M
                obj.weight{m} = 0.01 * randn(size(obj.weight{m})); % 将权值初始化为0附近的随机数
                obj.bias{m} = zeros(size(obj.bias{m}));            % 将偏置值初始化为0
            end
        end
        
        function obj = train(obj,minibatchs,learn_rate_min,learn_rate_max,max_it)
            minibatch_num = length(minibatchs);
            ob_window_size = minibatch_num;
            ob = ML.Observer('均方误差',1,ob_window_size,'xxx');
            
            learn_rate = learn_rate_max; % 初始化学习速度为最大学习速度
            error_list = zeros(1,ob_window_size);
            for minibatch_idx = 1:minibatch_num  % 初始化误差列表的移动平均值
                labels = minibatchs{minibatch_idx}.labels;
                points = minibatchs{minibatch_idx}.points;
                error = sum(sqrt(sum((labels - obj.do(points)).^2,2)));
                error_list(minibatch_idx) = error;
            end
            error_ave_old = mean(error_list);
            ob = ob.initialize(error_ave_old);
            
            for it = 0:max_it
                minibatch_idx = mod(it,minibatch_num) + 1;
                labels = minibatchs{minibatch_idx}.labels;
                points = minibatchs{minibatch_idx}.points; 
                % minibatch_size = size(points,2);
                error = sum(sqrt(sum((labels - obj.do(points)).^2,2)));
                error_list(minibatch_idx) = error;
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
                
                delta = obj.BP(points(:,1),labels(:,1));
%                 delta_w = 2 * delta * points' / minibatch_size;
%                 delta_b = 2 * delta * ones(minibatch_size,1) / minibatch_size;
%                 obj.weight = obj.weight + learn_rate * delta_w;
%                 obj.bias   = obj.bias   + learn_rate * delta_b;
            end
        end
        
        function [y,n,a] = do(obj,x)
            % 多层感知器的计算过程
            % y 是最后的输出
            % n 是每一层的局部诱导域
            % a 是每一层的输出
            
            M = length(obj.bias);           % 得到层数
            n = cell(1,M); a = cell(1,M);   % n是每一层的局部诱导域，a是每一层的输出
          
            for m = 1:M
                n{m} = obj.weight{m} * x + repmat(obj.bias{m},1,size(x,2));
                a{m} = ML.sigmoid(n{m});
                x = a{m};
            end
            
            y = a{M};
        end
    end
    
    methods (Access = private)
        function delta = BP(obj,point,label)
            % 反向传播算法
            M = length(obj.bias); % 得到层数

            [~,~,a] = obj.do(point); % 首先执行正向计算,并记录每一层的输出
            s{M} = diag(2 * (label - a{M}) .* a{M} .* (a{M} - 1)); % 计算顶层的敏感性
            
            for m = (M-1):-1:1  % 反向传播敏感性
                s{m} = s{m+1} * obj.weight{m+1} * diag(a{m} .* (1 - a{m})); 
            end
            
            for m = 1:M  % 计算梯度
                if m == 1 
                    delta{m}.weight = s{m} * repmat(point',size(a{m}));
                    delta{m}.bias = 1;
                else
                    delta{m}.weight = s{m} * repmat(a{m-1}',size(a{m-1},1),1);
                    delta{m}.bias = 1;
                end
            end
        end
    end
    
    methods(Static)
        function [perception,e] = unit_test()
            clear all;
            close all;
            [train_images,train_labels,test_images,test_labels] = ML.import_mnist('./+ML/mnist.mat');
            [D,minibatch_size,minibatch_num] = size(train_images); K = 10;
            for minibatch_idx = 1:minibatch_num
                mnist{minibatch_idx}.labels = zeros(K,minibatch_size);
                I = sub2ind([K,minibatch_size],1+train_labels(:,minibatch_idx),[1:minibatch_size]');
                mnist{minibatch_idx}.labels(I) = 1;
                mnist{minibatch_idx}.points = train_images(:,:,minibatch_idx);
            end
            
            configure = [D,500,600,2000,K];
            
            perception = ML.Perception(configure);
            perception = perception.initialize();
            perception = perception.train(mnist,1e-6,0.1,1e6);
            
            save('perception.mat','perception');
            
%             y = perception.classify(test_images);
%             e = sum(y~=test_labels') / length(y);
        end
    end
end


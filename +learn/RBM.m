classdef RBM
    %RestrictedBoltzmannMachine 约束玻尔兹曼机
    %   
    
    properties
        num_hidden; % 隐藏神经元的个数
        num_visual; % 可见神经元的个数
        weight;     % 权值矩阵(num_hidden * num_visual)
        hidden_bias; % 隐藏神经元的偏置
        visual_bias; % 可见神经元的偏置
    end
    
    methods
        function obj = RBM(num_visual,num_hidden) % 构造函数
            obj.num_hidden = num_hidden;
            obj.num_visual = num_visual;
            obj.weight = zeros(obj.num_hidden,obj.num_visual);
            obj.hidden_bias = zeros(obj.num_hidden,1);
            obj.visual_bias = zeros(obj.num_visual,1);
        end
    end
    
    methods
        function obj = construct(obj,weight,visual_bias,hidden_bias)
            %construct 使用权值、隐神经元偏置、显神经元偏置直接构建RBM
            [obj.num_hidden, obj.num_visual] = size(weight);
            obj.weight = weight;
            obj.hidden_bias = hidden_bias;
            obj.visual_bias = visual_bias;
        end
        
        function obj = initialize(obj,minibatchs) 
            %INTIALIZE 初始化权值矩阵为0附近的小随机数，初始化显层神经元的偏置为先验概率，初始化隐层神经元的偏置为0.   
            [D,S,M] = size(minibatchs);
            minibatchs = reshape(minibatchs,D,[]);
            obj = obj.initialize_weight(minibatchs);
        end

        function obj = pretrain(obj,minibatchs,learn_rate,weight_cost,max_it) 
            %pretrain 对权值进行预训练
            % 使用CD1快速算法对权值进行预训练
            
            [D,S,M] = size(minibatchs); % 得到minibatch的个数
            ob_window_size = M;     % 设定观察窗口的大小为
            ob_var_num = 1;                     % 设定观察变量的个数
            ob = learn.Observer('重建误差',ob_var_num,ob_window_size,'xxx'); %初始化观察者，观察重建误差
            
            % 初始化velocity变量
            v_weight = zeros(size(obj.weight));
            v_h_bias = zeros(size(obj.hidden_bias));
            v_v_bias = zeros(size(obj.visual_bias));
            
            % 初始化动量倍率为0.5
            momentum = 0.5;
            
            r_error_list = zeros(1,ob_window_size);
            for idx = 1:M  % 初始化重建误差列表的移动平均值
                minibatch = minibatchs(:,:,idx);
                [~, ~, ~, r_error] = obj.CD1(minibatch);
                r_error_list(idx) = r_error;
            end
            r_error_ave_old = mean(r_error_list);
            ob = ob.initialize(r_error_ave_old);
            
            learn_rate_min = min(learn_rate);
            learn_rate     = max(learn_rate); %初始化学习速度
            
            for it = 0:max_it
                minibatch_idx = mod(it,M)+1;  % 取一个minibatch
                minibatch = minibatchs(:,:,minibatch_idx);
                
                [d_weight, d_h_bias, d_v_bias, r_error] = obj.CD1(minibatch);
                r_error_list(minibatch_idx) = r_error;
                r_error_ave_new = mean(r_error_list);
                
                if minibatch_idx == M % 当所有的minibatch被轮讯了一篇的时候（到达观察窗口最右边的时候）
                    if r_error_ave_new > r_error_ave_old
                        learn_rate = learn_rate / 2;
                        if learn_rate < learn_rate_min
                            break;
                        end
                    end
                    r_error_ave_old = r_error_ave_new;
                end
                
                description = strcat('重建误差:',num2str(r_error_ave_new));
                description = strcat(description,strcat('迭代次数:',num2str(it)));
                description = strcat(description,strcat('学习速度:',num2str(learn_rate)));
                % disp(description);
                ob = ob.showit(r_error_ave_new,description);
                
                momentum = min([momentum * 1.01,0.9]); % 动量倍率最大为0.9，初始值为0.5，大约迭代60步之后动量倍率达到0.9。
                
                v_weight = momentum * v_weight + learn_rate * (d_weight - weight_cost * obj.weight);
                v_h_bias = momentum * v_h_bias + learn_rate * d_h_bias;
                v_v_bias = momentum * v_v_bias + learn_rate * d_v_bias;
                
                obj.weight      = obj.weight      + v_weight;
                obj.hidden_bias = obj.hidden_bias + v_h_bias;
                obj.visual_bias = obj.visual_bias + v_v_bias;
            end
        end
        
        function h_state = posterior_sample(obj,v_state)
            % posterior_sample 计算后验概率采样
            % 在给定显层神经元取值的情况下，对隐神经元进行抽样
            h_state = learn.sample(obj.posterior(v_state));
        end
        
        function h_field = posterior(obj,v_state) 
            %POSTERIOR 计算后验概率
            % 在给定显层神经元取值的情况下，计算隐神经元的激活概率
            h_field = learn.sigmoid(obj.foreward(v_state));
        end
        
        function v_state = likelihood_sample(obj,h_state) 
            % likelihood_sample 计算似然概率采样
            % 在给定隐层神经元取值的情况下，对显神经元进行抽样
            v_state = learn.sample(obj.likelihood(h_state));
        end
        
        function v_field = likelihood(obj,h_state) 
            % likelihood 计算似然概率
            % 在给定隐层神经元取值的情况下，计算显神经元的激活概率
            v_field = learn.sigmoid(obj.backward(h_state));
        end
        
        function y = foreward(obj,x)
            y = obj.weight * x + repmat(obj.hidden_bias,1,size(x,2));
        end
        
        function x = backward(obj,y)
            x = obj.weight'* y + repmat(obj.visual_bias,1,size(y,2));
        end
        
        function y = reconstruct(obj,x)
            N = size(x,2);
            h_bias = repmat(obj.hidden_bias,1,N);
            v_bias = repmat(obj.visual_bias,1,N);
            
            h_field_0 = learn.sigmoid(obj.weight * x + h_bias);
            h_state_0 = learn.sample(h_field_0);
            y = learn.sigmoid(obj.weight'* h_state_0 + v_bias);
        end
    end
    
    methods (Access = private)
        function [d_weight,d_h_bias,d_v_bias,r_error] = CD1(obj, minibatch)
            % 使用Contrastive Divergence 1 (CD1)方法对约束玻尔兹曼RBM机进行快速训练
            % 输入：
            %   minibatch，一组训练数据，一列代表一个训练样本，列数表示训练样本的个数
            % 输出
            %   d_weight, 权值矩阵的迭代差值.
            %   d_h_bias, 隐藏神经元偏置值的迭代差值.
            %   d_v_bias, 可见神经元偏置值的迭代差值.
            %   r_error,  重建误差值
            
            N = size(minibatch,2); % 训练样本的个数
            h_bias = repmat(obj.hidden_bias,1,N);
            v_bias = repmat(obj.visual_bias,1,N);
            
            h_field_0 = learn.sigmoid(obj.weight * minibatch + h_bias);
            h_state_0 = learn.sample(h_field_0);
            v_field_1 = learn.sigmoid(obj.weight'* h_state_0 + v_bias);
            %v_state_1 = learn.sample(v_field_1);
            v_state_1 = v_field_1; % 此处不做采样，可增加精度，与Hinton的程序相同
            h_field_1 = learn.sigmoid(obj.weight * v_state_1 + h_bias);
            
            r_error =  sum(sum((v_field_1 - minibatch).^2)) / N; %计算在整个minibatch上的平均重建误差
            
            d_weight = (h_field_0 * minibatch' - h_field_1 * v_state_1') / N;
            d_h_bias = (h_field_0 - h_field_1) * ones(N,1) / N;
            d_v_bias = (minibatch - v_state_1) * ones(N,1) / N;
        end
        
        function obj = initialize_weight(obj,train_data)
            %INTIALIZE 初始化权值矩阵为0附近的小随机数，初始化显层神经元的偏置为先验概率，初始化隐层神经元的偏置为0.
            obj.weight = 0.01 * randn(size(obj.weight));
            obj.visual_bias = mean(train_data,2);
            obj.visual_bias = log(obj.visual_bias./(1-obj.visual_bias));
            obj.visual_bias(obj.visual_bias < -100) = -100;
            obj.visual_bias(obj.visual_bias > +100) = +100;
            obj.hidden_bias = zeros(size(obj.hidden_bias));
        end
    end
    
    methods(Static)
        function [rbm,e] = unit_test()
            clear all;
            close all;
            [data,~,~,~] = learn.import_mnist('./+learn/mnist.mat');
            [D,S,M] = size(data); N = S * M;
     
            rbm = learn.RBM(D,500);
            rbm = rbm.initialize(data);
            rbm = rbm.pretrain(data,[1e-6,1e-1],1e-4,1e5);
            
            save('rbm.mat','rbm');
         
            data = reshape(data,D,[]);
            recon_data = rbm.reconstruct(data);
            e = sum(sum((255*recon_data - 255*data).^2)) / N;
        end
    end
    
end


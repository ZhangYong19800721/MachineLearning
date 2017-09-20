classdef RBM
    %RestrictedBoltzmannMachine 约束玻尔兹曼机
    %   
    
    properties
        num_hidden; % 隐藏神经元的个数
        num_visual; % 可见神经元的个数
        weight_v2h;  % 权值矩阵(num_hidden * num_visual)
        weight_h2v;  % 权值矩阵(num_hidden * num_visual)
        hidden_bias; % 隐藏神经元的偏置
        visual_bias; % 可见神经元的偏置
    end
    
    methods
        function obj = RBM(num_visual,num_hidden) % 构造函数
            obj.num_hidden  = num_hidden;
            obj.num_visual  = num_visual;
            obj.weight_v2h  = zeros(obj.num_hidden,obj.num_visual);
            obj.weight_h2v  = [];
            obj.hidden_bias = zeros(obj.num_hidden,1);
            obj.visual_bias = zeros(obj.num_visual,1);
        end
    end
    
    methods
        function obj = construct(obj,weight_v2h,weight_h2v,visual_bias,hidden_bias)
            %construct 使用权值、隐神经元偏置、显神经元偏置直接构建RBM
            [obj.num_hidden, obj.num_visual] = size(weight_v2h);
            obj.weight_v2h = weight_v2h;         
            obj.weight_h2v = weight_h2v;
            obj.hidden_bias = hidden_bias;
            obj.visual_bias = visual_bias;
        end
        
        function obj = initialize(obj,minibatchs) 
            %INTIALIZE 初始化权值矩阵为0附近的小随机数，初始化显层神经元的偏置为先验概率，初始化隐层神经元的偏置为0.   
            [D,S,M] = size(minibatchs);
            minibatchs = reshape(minibatchs,D,[]);
            obj = obj.initialize_weight(minibatchs);
        end

        function obj = pretrain(obj,minibatchs,parameters) 
            %pretrain 对权值进行预训练
            % 使用CD1快速算法对权值进行预训练
            
            [D,S,M] = size(minibatchs); % 得到minibatch的个数
            
            % 初始化velocity变量
            v_weight = zeros(size(obj.weight_v2h));
            v_h_bias = zeros(size(obj.hidden_bias));
            v_v_bias = zeros(size(obj.visual_bias));
            
            % 初始化动量倍率为0.5
            momentum = 0.5;
            
            recon_error_list = zeros(1,M);
            for m = 1:M  % 初始化重建误差列表的移动平均值
                minibatch = minibatchs(:,:,m);
                recon_data = obj.rebuild(minibatch);
                recon_error = sum(sum((recon_data - minibatch).^2)) / S;
                recon_error_list(m) = recon_error;
            end
            recon_error_ave_old = mean(recon_error_list);
            
            learn_rate_min = min(parameters.learn_rate);
            learn_rate     = max(parameters.learn_rate); %初始化学习速度
            
            for it = 0:parameters.max_it
                minibatch_idx = mod(it,M)+1;  % 取一个minibatch
                minibatch = minibatchs(:,:,minibatch_idx);
                
                [d_weight, d_h_bias, d_v_bias, recon_error] = obj.CD1(minibatch);
                recon_error_list(minibatch_idx) = recon_error;
                recon_error_ave_new = mean(recon_error_list);
                
                if minibatch_idx == M % 当所有的minibatch被轮讯了一篇的时候（到达观察窗口最右边的时候）
                    if recon_error_ave_new > recon_error_ave_old
                        learn_rate = learn_rate / 2;
                        if learn_rate < learn_rate_min
                            break;
                        end
                    end
                    recon_error_ave_old = recon_error_ave_new;
                end
                
                disp(sprintf('重建误差:%f 迭代次数:%d 学习速度:%f',recon_error_ave_new,it,learn_rate));
                
                momentum = min([momentum * 1.01,0.9]); % 动量倍率最大为0.9，初始值为0.5，大约迭代60步之后动量倍率达到0.9。
                
                v_weight = momentum * v_weight + learn_rate * (d_weight - parameters.weight_cost * obj.weight_v2h);
                v_h_bias = momentum * v_h_bias + learn_rate * d_h_bias;
                v_v_bias = momentum * v_v_bias + learn_rate * d_v_bias;
                
                obj.weight_v2h  = obj.weight_v2h  + v_weight;
                obj.hidden_bias = obj.hidden_bias + v_h_bias;
                obj.visual_bias = obj.visual_bias + v_v_bias;
            end
        end
        
        function h_state = posterior_sample(obj,v_state)
            % posterior_sample 计算后验概率采样
            % 在给定显层神经元取值的情况下，对隐神经元进行抽样
            h_state = learn.tools.sample(obj.posterior(v_state));
        end
        
        function h_field = posterior(obj,v_state) 
            %POSTERIOR 计算后验概率
            % 在给定显层神经元取值的情况下，计算隐神经元的激活概率
            h_field = learn.tools.sigmoid(obj.foreward(v_state));
        end
        
        function v_state = likelihood_sample(obj,h_state) 
            % likelihood_sample 计算似然概率采样
            % 在给定隐层神经元取值的情况下，对显神经元进行抽样
            v_state = learn.sample(obj.likelihood(h_state));
        end
        
        function v_field = likelihood(obj,h_state) 
            % likelihood 计算似然概率
            % 在给定隐层神经元取值的情况下，计算显神经元的激活概率
            v_field = learn.tools.sigmoid(obj.backward(h_state));
        end
        
        function y = foreward(obj,x)
            y = obj.weight_v2h * x + repmat(obj.hidden_bias,1,size(x,2));
        end
        
        function x = backward(obj,y)
            if isempty(obj.weight_h2v)
                x = obj.weight_v2h'* y + repmat(obj.visual_bias,1,size(y,2));
            else
                x = obj.weight_h2v'* y + repmat(obj.visual_bias,1,size(y,2));
            end
        end
        
        function y = rebuild(obj,x)
            z = obj.posterior_sample(x);
            y = obj.likelihood(z);
        end
        
        function obj = weightsync(obj)
            obj.weight_h2v = obj.weight_v2h;
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
            
            h_field_0 = learn.tools.sigmoid(obj.weight_v2h * minibatch + h_bias);
            h_state_0 = learn.tools.sample(h_field_0);
            v_field_1 = learn.tools.sigmoid(obj.weight_v2h'* h_state_0 + v_bias);
            %v_state_1 = learn.sample(v_field_1); 
            v_state_1 = v_field_1; 
            h_field_1 = learn.tools.sigmoid(obj.weight_v2h * v_state_1 + h_bias);
            r_error =  sum(sum((v_field_1 - minibatch).^2)) / N; %计算在整个minibatch上的平均重建误差
            d_weight = (h_field_0 * minibatch' - h_field_1 * v_state_1') / N;
            d_h_bias = (h_field_0 - h_field_1) * ones(N,1) / N;
            d_v_bias = (minibatch - v_state_1) * ones(N,1) / N;
        end
        
        function obj = initialize_weight(obj,train_data)
            %INTIALIZE 初始化权值矩阵为0附近的小随机数，初始化显层神经元的偏置为先验概率，初始化隐层神经元的偏置为0.
            obj.weight_v2h = 0.01 * randn(size(obj.weight_v2h));
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
            
            parameters.learn_rate = [1e-6,1e-2];
            parameters.weight_cost = 1e-4;
            parameters.max_it = 1e5;
            rbm = rbm.pretrain(data,parameters);
            
            save('rbm.mat','rbm');
         
            data = reshape(data,D,[]);
            recon_data = rbm.reconstruct(data);
            e = sum(sum((255*recon_data - 255*data).^2)) / N;
        end
    end
    
end


classdef SoftmaxRBM
    %Softmax Restricted Boltzmann Machine 带Softmax神经元的约束玻尔兹曼机
    %   
    
    properties
        num_hidden;  % 隐藏神经元的个数
        num_visual;  % 可见神经元的个数
        num_softmax; % softmax神经元的个数
        weight_v2h;  % 权值矩阵(num_hidden * num_visual)
        weight_s2h;  % 权值矩阵(num_hidden * num_softmax)
        weight_h2v;  % 权值矩阵(num_hidden * num_visual)
        weight_h2s;  % 权值矩阵(num_hidden * num_softmax)
        hidden_bias; % 隐藏神经元的偏置
        visual_bias; % 可见神经元的偏置
        softmax_bias;% softmax神经元的偏置
    end
    
    methods
        function obj = SoftmaxRBM(num_softmax,num_visual,num_hidden) % 构造函数
            obj.num_hidden = num_hidden;
            obj.num_visual = num_visual;
            obj.num_softmax = num_softmax;
            obj.weight_v2h = zeros(obj.num_hidden,obj.num_visual);
            obj.weight_s2h = zeros(obj.num_hidden,obj.num_softmax);
            obj.weight_h2v = [];
            obj.weight_h2s = [];
            obj.hidden_bias = zeros(obj.num_hidden,1);
            obj.visual_bias = zeros(obj.num_visual,1);
            obj.softmax_bias = zeros(obj.softmax_bias,1);
        end
    end
    
    methods
        function obj = construct(obj,weight_s2h,weight_v2h,weight_h2s,weight_h2v,softmax_bias,visual_bias,hidden_bias)
            %construct 使用softmax神经元个数、权值、显神经元偏置、隐神经元偏置直接构建RBM
            obj.num_softmax = length(softmax_bias);
            obj.num_visual  = length(visual_bias);
            obj.num_hidden  = length(hidden_bias);
            
            obj.weight_s2h  = weight_s2h;
            obj.weight_v2h  = weight_v2h;
            obj.weight_h2s  = weight_h2s;
            obj.weight_h2v  = weight_h2v;
            
            obj.softmax_bias = softmax_bias;
            obj.hidden_bias  = hidden_bias;
            obj.visual_bias  = visual_bias;
        end
        
        function obj = initialize(obj,minibatchs,labels) 
            %initialize 基于训练数据（由多个minibatch组成的训练数据集合）初始化权值矩阵，隐神经元偏置和显神经元偏置
            %           minibatchs 是一个元胞数组。
            
            [D,~,~] = size(minibatchs);
            [K,~,~] = size(labels);
            minibatchs = reshape(minibatchs,D,[]);
            labels     = reshape(labels    ,K,[]);
            obj = obj.initialize_weight(minibatchs,labels);
        end

        function obj = pretrain(obj,minibatchs,labels,parameters) 
            %pretrain 对权值进行预训练
            % 使用CD1快速算法对权值进行预训练
            
            [D,S,M] = size(minibatchs); % 得到minibatch的个数
            ob = learn.Observer('重建误差',1,M); %初始化观察者，观察重建误差
            
            % 初始化velocity变量
            inc_w_s2h  = zeros(size(obj.weight_s2h));
            inc_w_v2h  = zeros(size(obj.weight_v2h));
            inc_h_bias = zeros(size(obj.hidden_bias));
            inc_s_bias = zeros(size(obj.softmax_bias));
            inc_v_bias = zeros(size(obj.visual_bias));
            
            % 初始化动量倍率为0.5
            momentum = 0.5;
            
            recon_error_list = zeros(1,M);
            for m = 1:M  % 初始化重建误差列表的移动平均值
                label = labels(:,:,m);
                minibatch = minibatchs(:,:,m);
                [~,~,~,~,~,recon_error] = obj.CD1(minibatch,label);
                recon_error_list(m) = recon_error;
            end
            recon_error_ave_old = mean(recon_error_list);
            ob = ob.initialize(recon_error_ave_old);
            
            learn_rate_min = min(parameters.learn_rate);
            learn_rate     = max(parameters.learn_rate); %初始化学习速度
            
            for it = 0:parameters.max_it
                m = mod(it,M)+1;  % 取一个minibatch
                label = labels(:,:,m);
                minibatch = minibatchs(:,:,m);
                
                [d_w_s2h,d_w_v2h,d_h_bias,d_s_bias,d_v_bias,recon_error] = obj.CD1(minibatch,label);
                recon_error_list(m) = recon_error;
                recon_error_ave_new = mean(recon_error_list);
                
                if m == M % 当所有的minibatch被轮讯了一篇的时候（到达观察窗口最右边的时候）
                    if recon_error_ave_new > recon_error_ave_old
                        learn_rate = learn_rate / 2;
                        if learn_rate < learn_rate_min
                            break;
                        end
                    end
                    recon_error_ave_old = recon_error_ave_new;
                end
                
                description = strcat('重建误差:',num2str(recon_error_ave_new));
                description = strcat(description,strcat('迭代次数:',num2str(it)));
                description = strcat(description,strcat('学习速度:',num2str(learn_rate)));
                disp(description);
                % ob = ob.showit(recon_error_ave_new,description);
                
                momentum = min([momentum * 1.01,0.9]); % 动量倍率最大为0.9，初始值为0.5，大约迭代60步之后动量倍率达到0.9。
                inc_w_s2h  = momentum * inc_w_s2h  + learn_rate * (d_w_s2h - parameters.weight_cost * obj.weight_s2h);
                inc_w_v2h  = momentum * inc_w_v2h  + learn_rate * (d_w_v2h - parameters.weight_cost * obj.weight_v2h);
                inc_h_bias = momentum * inc_h_bias + learn_rate * d_h_bias;
                inc_s_bias = momentum * inc_s_bias + learn_rate * d_s_bias;
                inc_v_bias = momentum * inc_v_bias + learn_rate * d_v_bias;
                
                obj.weight_s2h   = obj.weight_s2h   + inc_w_s2h;
                obj.weight_v2h   = obj.weight_v2h   + inc_w_v2h;
                obj.hidden_bias  = obj.hidden_bias  + inc_h_bias;
                obj.softmax_bias = obj.softmax_bias + inc_s_bias;
                obj.visual_bias  = obj.visual_bias  + inc_v_bias;
            end
        end
        
        function h_state = posterior_sample(obj,s_state,v_state)
            % posterior_sample 计算后验概率采样
            % 在给定显层神经元取值的情况下，对隐神经元进行抽样
            h_state = learn.sample(obj.posterior(s_state,v_state));
        end
        
        function h_field = posterior(obj,s_state,v_state) 
            %POSTERIOR 计算后验概率
            % 在给定显层神经元取值的情况下，计算隐神经元的激活概率
            h_field = learn.sigmoid(obj.foreward(s_state,v_state));
        end
        
        function [s_state,v_state] = likelihood_sample(obj,h_state) 
            % likelihood_sample 计算似然概率采样
            % 在给定隐层神经元取值的情况下，对显神经元进行抽样
            [s_field,v_field] = obj.likelihood(h_state);
            s_state = learn.sample_softmax(s_field);
            v_state = learn.sample(v_field);
        end
        
        function [s_field,v_field] = likelihood(obj,h_state) 
            % likelihood 计算似然概率
            % 在给定隐层神经元取值的情况下，计算显神经元的激活概率
            [s,x] = obj.backward(h_state);
            s_field = learn.softmax(s);
            v_field = learn.sigmoid(x);
        end
        
        function y = foreward(obj,s,x)
            y = [obj.weight_s2h obj.weight_v2h] * [s;x] + repmat(obj.hidden_bias,1,size(x,2));
        end
        
        function [s,x] = backward(obj,y)
            if isempty(obj.weight_h2v)
                s = obj.weight_s2h'* y + repmat(obj.softmax_bias,1,size(y,2));
                x = obj.weight_v2h'* y + repmat(obj.visual_bias, 1,size(y,2));
            else
                s = obj.weight_h2s'* y + repmat(obj.softmax_bias,1,size(y,2));
                x = obj.weight_h2v'* y + repmat(obj.visual_bias, 1,size(y,2));
            end
        end
        
        function [c,y] = rebuild(obj,s,x)
            z = obj.posterior_sample(s,x);
            [c,y] = obj.likelihood(z);
        end
        
        function obj = weightsync(obj)
            obj.weight_h2s = obj.weight_s2h;
            obj.weight_h2v = obj.weight_v2h;
        end
        
        function y = classify(obj,x)
            %classify 给定数据点，计算该数据的分类
            %
            N = size(x,2); % 数据样本点的个数
            E = inf * ones(obj.num_softmax,N);
                
            for n = 1:obj.num_softmax
                s = zeros(obj.num_softmax,N); s(n,:) = 1;
                E(n,:) = -obj.softmax_bias' * s - obj.visual_bias' * x - ... % 计算自由能量
                    sum(log(1 + exp(obj.weight_s2h * s + obj.weight_v2h * x + repmat(obj.hidden_bias,1,N))));
            end
            
            [~,y] = min(E);
            y = y - 1;
        end
    end
    
    methods (Access = private)
        function [d_w_s2h,d_w_v2h,d_h_bias,d_s_bias,d_v_bias,recon_error] = CD1(obj, minibatch, label)
            % 使用Contrastive Divergence 1 (CD1)方法对约束玻尔兹曼RBM机进行快速训练，该RBM包含Softmax神经元。
            % 输入：
            %   minibatch，一组训练数据，一列代表一个训练样本，列数表示训练样本的个数
            % 输出:
            %   d_weight, 权值矩阵的迭代差值.
            %   d_h_bias, 隐藏神经元偏置值的迭代差值.
            %   d_v_bias, 可见神经元偏置值的迭代差值.
            %   r_error,  重建误差值
        
            N = size(minibatch,2); % 训练样本的个数
            
            h_field_0 = obj.posterior(label, minibatch);
            h_state_0 = learn.sample(h_field_0);
            [s_field_1,v_field_1] = obj.likelihood(h_state_0);
            s_state_1 = learn.sample_softmax(s_field_1);
            v_state_1 = learn.sample(v_field_1);
            h_field_1 = obj.posterior(s_state_1,v_state_1); 
            
            recon_error =  sum(sum(([s_field_1;v_field_1] - [label;minibatch]).^2)) / N; %计算在整个train_data上的平均reconstruction error
            
            d_w_s2h  = (h_field_0 * label'     - h_field_1 * s_state_1') / N;
            d_w_v2h  = (h_field_0 * minibatch' - h_field_1 * v_state_1') / N;
            d_h_bias = (h_state_0 - h_field_1) * ones(N,1) / N;
            d_s_bias = (label     - s_field_1) * ones(N,1) / N;
            d_v_bias = (minibatch - v_field_1) * ones(N,1) / N;
        end
        
        function obj = initialize_weight(obj,train_data,train_label)
            %INTIALIZE 初始化权值矩阵为0附近的小随机数，初始化显层神经元的偏置为先验概率，初始化隐层神经元的偏置为0.
            obj.weight_s2h = 0.01 * randn(size(obj.weight_s2h));
            obj.weight_v2h = 0.01 * randn(size(obj.weight_v2h));
            
            obj.softmax_bias = mean(train_label,2);
            obj.visual_bias  = mean(train_data, 2);
            
            obj.softmax_bias = log(obj.softmax_bias ./ (1-obj.softmax_bias));
            obj.visual_bias  = log(obj.visual_bias  ./ (1-obj.visual_bias));
            
            obj.softmax_bias(obj.softmax_bias < -100) = -100;
            obj.softmax_bias(obj.softmax_bias > +100) = +100;
            
            obj.visual_bias(obj.visual_bias < -100) = -100;
            obj.visual_bias(obj.visual_bias > +100) = +100;
            
            obj.hidden_bias = zeros(size(obj.hidden_bias));
        end
    end
    
    methods(Static)
        function [srbm,e] = unit_test()
            clear all;
            close all;
            [data,~,test_data,test_label] = learn.import_mnist('./+learn/mnist.mat');
            [D,S,M] = size(data); N = S * M; K = 10;
            label = eye(10); label = repmat(label,1,10,M);
     
            srbm = learn.SoftmaxRBM(K,D,2000);
            srbm = srbm.initialize(data,label);
            
            parameters.learn_rate = [1e-6,1e-2];
            parameters.weight_cost = 1e-4;
            parameters.max_it = 1e6;
            srbm = srbm.pretrain(data,label,parameters);
            
            save('srbm.mat','srbm');
            % load('srbm.mat');
         
            y = srbm.classify(test_data);
            e = sum(y~=test_label') / length(test_label);
        end
    end
end


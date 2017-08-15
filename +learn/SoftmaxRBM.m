classdef SoftmaxRBM
    %Softmax Restricted Boltzmann Machine 带Softmax神经元的约束玻尔兹曼机
    %   
    
    properties
        num_hidden;  % 隐藏神经元的个数
        num_visual;  % 可见神经元的个数
        num_softmax; % softmax神经元的个数
        weight;      % 权值矩阵(num_hidden * num_visual)
        hidden_bias; % 隐藏神经元的偏置
        visual_bias; % 可见神经元的偏置
    end
    
    methods
        function obj = SoftmaxRBM(num_softmax,num_visual,num_hidden) % 构造函数
            obj.num_hidden = num_hidden;
            obj.num_visual = num_softmax + num_visual;
            obj.num_softmax = num_softmax;
            obj.weight = zeros(obj.num_hidden,obj.num_visual);
            obj.hidden_bias = zeros(obj.num_hidden,1);
            obj.visual_bias = zeros(obj.num_visual,1);
        end
    end
    
    methods
        function obj = construct(num_softmax,obj,weight,visual_bias,hidden_bias)
            %construct 使用softmax神经元个数、权值、显神经元偏置、隐神经元偏置直接构建RBM
            [obj.num_hidden, obj.num_visual] = size(weight);
            obj.num_softmax = num_softmax;
            obj.weight = weight;
            obj.hidden_bias = hidden_bias;
            obj.visual_bias = visual_bias;
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
            v_weight      = zeros(size(obj.weight));
            v_h_bias = zeros(size(obj.hidden_bias));
            v_v_bias = zeros(size(obj.visual_bias));
            
            % 初始化动量倍率为0.5
            momentum = 0.5;
            
            recon_error_list = zeros(1,M);
            for m = 1:M  % 初始化重建误差列表的移动平均值
                label = labels(:,:,m);
                minibatch = minibatchs(:,:,m);
                [~, ~, ~, recon_error] = obj.CD1(minibatch,label);
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
                
                [d_weight, d_h_bias, d_v_bias, recon_error] = obj.CD1(minibatch,label);
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
                % ob = ob.showit(r_error_ave_new,description);
                
                momentum = min([momentum * 1.01,0.9]); % 动量倍率最大为0.9，初始值为0.5，大约迭代60步之后动量倍率达到0.9。
                v_weight = momentum * v_weight + learn_rate * (d_weight - parameters.weight_cost * obj.weight);
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
            N = size(v_state,2);
            h_field = learn.sigmoid(obj.weight * v_state + repmat(obj.hidden_bias,1,N));
        end
        
        function [s_state,v_state] = likelihood_sample(obj,h_state) 
            % likelihood_sample 计算似然概率采样
            % 在给定隐层神经元取值的情况下，对显神经元进行抽样
            v_field = obj.likelihood(h_state);
            s_state = learn.sample_softmax(v_field(1:obj.num_softmax,:));
            v_state = learn.sample(v_field((obj.num_softmax+1):obj.num_visual,:));
        end
        
        function [s_field,v_field] = likelihood(obj,h_state) 
            % likelihood 计算似然概率
            % 在给定隐层神经元取值的情况下，计算显神经元的激活概率
            N = size(h_state,2);
            v_sigma = obj.weight'* h_state + repmat(obj.visual_bias,1,N);
            s_field = learn.softmax(v_sigma(1:obj.num_softmax,:));
            v_field = learn.sigmoid(v_sigma((obj.num_softmax+1):obj.num_visual,:));
        end
        
        function y = classify(obj,x)
            %DISCRIMINATE 给定数据点，计算该数据的分类
            %
            N = size(x,2); % 数据样本点的个数
            y = -1 * ones(1,N);
     
            for n = 1:N
                v_state = x(:,n);
                min_energy = inf;
                
                for class_idx = 1:obj.num_softmax
                    s_state = zeros(obj.num_softmax,1); s_state(class_idx) = 1;
                    % 计算该显神经元对应的自由能量
                    free_energy = [s_state;v_state]' * obj.visual_bias + sum(log(1 + exp(obj.weight * [s_state;v_state] + obj.hidden_bias)));
                    free_energy = -1 * free_energy;
                    if free_energy < min_energy
                        min_energy = free_energy;
                        y(n) = class_idx - 1;
                    end
                end
            end
        end
    end
    
    methods (Access = private)
        function [d_weight,d_h_bias,d_v_bias,r_error] = CD1(obj, minibatch, labels)
            % 使用Contrastive Divergence 1 (CD1)方法对约束玻尔兹曼RBM机进行快速训练，该RBM包含Softmax神经元。
            % 输入：
            %   minibatch，一组训练数据，一列代表一个训练样本，列数表示训练样本的个数
            % 输出:
            %   d_weight, 权值矩阵的迭代差值.
            %   d_h_bias, 隐藏神经元偏置值的迭代差值.
            %   d_v_bias, 可见神经元偏置值的迭代差值.
            %   r_error,  重建误差值
        
            N = size(minibatch,2); % 训练样本的个数
            h_bias = repmat(obj.hidden_bias,1,N);
            v_bias = repmat(obj.visual_bias,1,N);
            
            h_field_0 = learn.sigmoid(obj.weight * [labels; minibatch] + h_bias);
            h_state_0 = learn.sample(h_field_0);
            v_sigma_1 = obj.weight'* h_state_0 + v_bias;
            s_field_1 = learn.softmax(v_sigma_1(1:obj.num_softmax,:));
            v_field_1 = learn.sigmoid(v_sigma_1((obj.num_softmax+1):obj.num_visual,:));
            s_state_1 = learn.sample_softmax(s_field_1);
            v_state_1 = learn.sample(v_field_1);
            h_field_1 = learn.sigmoid(obj.weight * [s_state_1;v_state_1] + h_bias);
            
            r_error =  sum(sum(([s_field_1;v_field_1] - [labels;minibatch]).^2)) / N; %计算在整个train_data上的平均reconstruction error
            
            d_weight = (h_field_0 * [labels;minibatch]' - h_field_1 * [s_state_1;v_state_1]') / N;
            d_h_bias = (h_state_0 - h_field_1) * ones(N,1) / N;
            d_v_bias = ([labels;minibatch] - [s_field_1;v_field_1]) * ones(N,1) / N;
        end
        
        function obj = initialize_weight(obj,train_data,train_label)
            %INTIALIZE 初始化权值矩阵为0附近的小随机数，初始化显层神经元的偏置为先验概率，初始化隐层神经元的偏置为0.
            obj.weight = 0.01 * randn(size(obj.weight));
            data = [train_label; train_data];
            obj.visual_bias = mean(data,2);
            obj.visual_bias = log(obj.visual_bias./(1-obj.visual_bias));
            obj.visual_bias(obj.visual_bias < -100) = -100;
            obj.visual_bias(obj.visual_bias > +100) = +100;
            obj.hidden_bias = zeros(size(obj.hidden_bias));
        end
    end
    
end


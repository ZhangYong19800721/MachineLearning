classdef GaussianRBM
    %Gaussian Bernoulli Restricted Boltzmann Machine 高斯伯努利约束玻尔兹曼机
    % 显层神经元为线性神经元加高斯噪声，隐层神经元为伯努利二值神经元
    
    properties
        num_hidden; % 隐藏神经元的个数
        num_visual; % 可见神经元的个数
        weight;     % 权值矩阵(num_hidden * num_visual)
        hidden_bias; % 隐藏神经元的偏置
        visual_bias; % 可见神经元的偏置
    end
    
    methods
        function obj = GaussianRBM(num_visual,num_hidden) % 构造函数
            obj.num_hidden = num_hidden;
            obj.num_visual = num_visual;
            obj.weight = zeros(obj.num_hidden,obj.num_visual);
            obj.hidden_bias = zeros(obj.num_hidden,1);
            obj.visual_bias = zeros(obj.num_visual,1);
        end
    end
    
    methods
%         function obj = construct(obj,weight,visual_bias,hidden_bias,visual_sgma)
%             %construct 使用权值、隐神经元偏置、显神经元偏置直接构建RBM
%             [obj.num_hidden, obj.num_visual] = size(weight);
%             obj.weight = weight;
%             obj.hidden_bias = hidden_bias;
%             obj.visual_bias = visual_bias;
%             obj.visual_sgma = visual_sgma;
%         end
        
        function obj = initialize(obj,minibatchs) 
            %initialize 基于训练数据（由多个minibatch组成的训练数据集合）初始化权值矩阵，隐神经元偏置和显神经元偏置
            %           minibatchs 是一个元胞数组。
           
            minibatch_num = length(minibatchs);
            minibatch_ave = zeros(size(minibatchs{1}));
            
            for n = 1:minibatch_num
                minibatch_ave = minibatch_ave + minibatchs{n};
            end
            
            minibatch_ave = minibatch_ave ./ minibatch_num;
            obj = obj.initialize_weight(minibatch_ave);
        end

        function obj = pretrain(obj,minibatchs,learn_rate_min,learn_rate_max,max_it) 
            %pretrain 对权值进行预训练
            % 使用CD1快速算法对权值进行预训练
            
            minibatch_num = length(minibatchs); % 得到minibatch的个数
            ob_window_size = minibatch_num;     % 设定观察窗口的大小为
            ob_var_num = 1;                     % 设定观察变量的个数
            ob = learn.Observer('重建误差',ob_var_num,ob_window_size,'xxx'); %初始化观察者，观察重建误差
            
            % 初始化velocity变量
            v_weight = zeros(size(obj.weight));
            v_h_bias = zeros(size(obj.hidden_bias));
            v_v_bias = zeros(size(obj.visual_bias));
            
            % 初始化动量倍率为0.5
            momentum = 0.5;
            
            r_error_list = zeros(1,ob_window_size);
            for idx = 1:minibatch_num  % 初始化重建误差列表的移动平均值
                minibatch = minibatchs{idx};
                [~, ~, ~, r_error] = obj.CD1(minibatch);
                r_error_list(idx) = r_error;
            end
            r_error_ave_old = mean(r_error_list);
            ob = ob.initialize(r_error_ave_old);
            
            learn_rate = learn_rate_max; %初始化学习速度
            
            for it = 0:max_it
                minibatch_idx = mod(it,minibatch_num)+1;  % 取一个minibatch
                minibatch = minibatchs{minibatch_idx};
                
                [d_weight, d_h_bias, d_v_bias, r_error] = obj.CD1(minibatch);
                r_error_list(minibatch_idx) = r_error;
                r_error_ave_new = mean(r_error_list);
                
                if minibatch_idx == minibatch_num % 当所有的minibatch被轮讯了一篇的时候（到达观察窗口最右边的时候）
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
                disp(description);
                ob = ob.showit(r_error_ave_new,description);
                
                momentum = min([momentum * 1.01,0.9]); % 动量倍率最大为0.9，初始值为0.5，大约迭代60步之后动量倍率达到0.9。
                v_weight = momentum * v_weight + learn_rate * d_weight;
                v_h_bias = momentum * v_h_bias + learn_rate * d_h_bias;
                v_v_bias = momentum * v_v_bias + learn_rate * d_v_bias;
                
                obj.weight      = obj.weight      + v_weight;
                obj.hidden_bias = obj.hidden_bias + v_h_bias;
                obj.visual_bias = obj.visual_bias + v_v_bias;
            end
        end
        
        function y = reconstruct(obj,x)
            N = size(x,2); % 样本的个数
            h_bias = repmat(obj.hidden_bias,1,N);
            v_bias = repmat(obj.visual_bias,1,N);
            
            h_field_0 = learn.sigmoid(obj.weight * x + h_bias);
            h_state_0 = learn.sample(h_field_0);
            y = obj.weight'* h_state_0 + v_bias;
        end
        
%         function h_state = posterior_sample(obj,v_state)
%             % posterior_sample 计算后验概率采样
%             % 在给定显层神经元取值的情况下，对隐神经元进行抽样
%             h_state = learn.sample(obj.posterior(v_state));
%         end
%         
%         function h_field = posterior(obj,v_state) 
%             %POSTERIOR 计算后验概率
%             % 在给定显层神经元取值的情况下，计算隐神经元的激活概率
%             h_field = learn.sigmoid(obj.foreward(v_state));
%         end
%         
%         function v_state = likelihood_sample(obj,h_state) 
%             % likelihood_sample 计算似然概率采样
%             % 在给定隐层神经元取值的情况下，对显神经元进行抽样
%             v_state = learn.sample(obj.likelihood(h_state));
%         end
%         
%         function v_field = likelihood(obj,h_state) 
%             % likelihood 计算似然概率
%             % 在给定隐层神经元取值的情况下，计算显神经元的激活概率
%             v_field = learn.sigmoid(obj.backward(h_state));
%         end
        
%         function y = foreward(obj,x)
%             y = obj.weight * x + repmat(obj.hidden_bias,1,size(x,2));
%         end
%         
%         function x = backward(obj,y)
%             x = obj.weight'* y + repmat(obj.visual_bias,1,size(y,2));
%         end
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
            v_field_1 = obj.weight'* h_state_0 + v_bias;
            v_state_1 = v_field_1 + randn(size(v_field_1));
            h_field_1 = learn.sigmoid(obj.weight * v_state_1 + h_bias);
            
            r_error =  sum(sum((v_field_1 - minibatch).^2)) / N; %计算在整个minibatch上的平均重建误差
            
            d_weight = (h_field_0 * minibatch' - h_field_1 * v_state_1') / N;
            d_h_bias = (h_state_0 - h_field_1) * ones(N,1) / N;
            d_v_bias = (minibatch - v_field_1) * ones(N,1) / N;
        end
        
        function obj = initialize_weight(obj,train_data)
            %INTIALIZE 初始化权值矩阵为0附近的小随机数，初始化显层神经元的偏置为先验概率，初始化隐层神经元的偏置为0.
            obj.weight = 0.01 * randn(size(obj.weight));
            obj.visual_bias = sum(train_data,2) / size(train_data,2);
            obj.hidden_bias = zeros(size(obj.hidden_bias));
        end
    end
    
    methods(Static)
        function [grbm,e] = unit_test()
            clear all;
            close all;
            rng(1);
            
            %[data,~,~,~] = learn.import_mnist('./+learn/mnist.mat'); data = 255 * data;
            
            D = 10; N = 1e5; S = 100; M = 1000;
            MU = 1:D; SIGMA = 10*rand(D); SIGMA = SIGMA * SIGMA';
            data = mvnrnd(MU,SIGMA,N)';
            X = data; AVE_X = repmat(mean(X,2),1,N);
            Z = double(X) - AVE_X;
            Y = Z*Z';
            [P,ZK] = eig(Y); 
            ZK=diag(ZK); 
            ZK(ZK<=0)=0;
            DK=ZK; DK(ZK>0)=1./(ZK(ZK>0)); 
            
            trwhitening =    sqrt(N-1)  * P * diag(sqrt(DK)) * P';
            dewhitening = (1/sqrt(N-1)) * P * diag(sqrt(ZK)) * P';
            
%             image = reshape(dewhitening * trwhitening * data(:,1,1),28,28)'; imshow(uint8(image));
%             image = reshape(dewhitening * trwhitening * data(:,2,1),28,28)'; imshow(uint8(image));
%             image = reshape(dewhitening * trwhitening * data(:,3,1),28,28)'; imshow(uint8(image));
%             image = reshape(dewhitening * trwhitening * data(:,4,1),28,28)'; imshow(uint8(image));
%             image = reshape(dewhitening * trwhitening * data(:,5,1),28,28)'; imshow(uint8(image));
%             image = reshape(dewhitening * trwhitening * data(:,6,1),28,28)'; imshow(uint8(image));
%             image = reshape(dewhitening * trwhitening * data(:,7,1),28,28)'; imshow(uint8(image));
%             image = reshape(dewhitening * trwhitening * data(:,8,1),28,28)'; imshow(uint8(image));
            
            data = trwhitening * Z;
            data = reshape(data,D,S,M); 
            
            for minibatch_idx = 1:M
                mnist{minibatch_idx} = data(:,:,minibatch_idx);
            end
            
            grbm = learn.GaussianRBM(D,500);
            grbm = grbm.initialize(mnist);
            grbm = grbm.pretrain(mnist,1e-6,1e-3,1e6);
            
            recon_data = dewhitening * grbm.reconstruct(trwhitening * Z) + AVE_X;
            
            image = reshape(recon_data(:,1),28,28)'; imshow(uint8(image));
            image = reshape(recon_data(:,2),28,28)'; imshow(uint8(image));
            image = reshape(recon_data(:,3),28,28)'; imshow(uint8(image));
            image = reshape(recon_data(:,4),28,28)'; imshow(uint8(image));
            image = reshape(recon_data(:,5),28,28)'; imshow(uint8(image));
            image = reshape(recon_data(:,6),28,28)'; imshow(uint8(image));
            image = reshape(recon_data(:,7),28,28)'; imshow(uint8(image));
            image = reshape(recon_data(:,8),28,28)'; imshow(uint8(image));
            image = reshape(recon_data(:,9),28,28)'; imshow(uint8(image));
            
            e = 1;
        end
    end
    
end


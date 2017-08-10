classdef GBRBM
    %Gaussian Bernoulli Restricted Boltzmann Machine 高斯伯努利约束玻尔兹曼机
    % 显层神经元为线性神经元加高斯噪声，隐层神经元为伯努利二值神经元
    
    properties
        num_hidden;  % 隐藏神经元的个数
        num_visual;  % 可见神经元的个数
        weight;      % 权值矩阵(num_hidden * num_visual)
        hidden_bias; % 隐藏神经元的偏置
        visual_bias; % 可见神经元的偏置
        visual_sgma; % 可见神经元加性高斯噪声标准差
    end
    
    methods
        function obj = GBRBM(num_visual,num_hidden) % 构造函数
            obj.num_hidden = num_hidden;
            obj.num_visual = num_visual;
            obj.weight = zeros(obj.num_hidden,obj.num_visual);
            obj.hidden_bias = zeros(obj.num_hidden,1);
            obj.visual_bias = zeros(obj.num_visual,1);
            obj.visual_sgma = ones(obj.num_visual,1);
        end
    end
    
    methods

        function obj = initialize(obj,minibatchs) 
            %initialize 基于训练数据（由多个minibatch组成的训练数据集合）初始化权值矩阵，隐神经元偏置和显神经元偏置
            %           minibatchs 是一个[D,S,M]维的数组。
           
            [D,S,M] = size(minibatchs);
            minibatchs = reshape(minibatchs,D,[]);
            obj = obj.initialize_weight(minibatchs);
        end

        function obj = pretrain(obj,minibatchs,learn_rate,learn_sgma,max_it) 
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
            v_v_sgma = zeros(size(obj.visual_sgma));
            
            % 初始化动量倍率为0.5
            momentum = 0.5;
            
            r_error_list = zeros(1,ob_window_size);
            for idx = 1:M  % 初始化重建误差列表的移动平均值
                minibatch = minibatchs(:,:,idx);
                [~, ~, ~, ~, r_error] = obj.CD1(minibatch);
                r_error_list(idx) = r_error;
            end
            r_error_ave_old = mean(r_error_list);
            ob = ob.initialize(r_error_ave_old);
            
            learn_rate_min = min(learn_rate);
            learn_rate     = max(learn_rate);  %初始化学习速度
            
            for it = 0:max_it
                minibatch_idx = mod(it,M)+1;  % 取一个minibatch
                minibatch = minibatchs(:,:,minibatch_idx);
                
                [d_weight, d_h_bias, d_v_bias, d_v_sgma, r_error] = obj.CD1(minibatch);
                r_error_list(minibatch_idx) = r_error;
                r_error_ave_new = mean(r_error_list);
                
                if minibatch_idx == M % 当所有的minibatch被轮讯了一遍的时候（到达观察窗口最右边的时候）
                    if r_error_ave_new > r_error_ave_old
                        learn_rate = learn_rate / 5;
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
                weigth_cost = 1e-4;
                v_weight = momentum * v_weight + learn_rate * (d_weight - weigth_cost * obj.weight);
                v_h_bias = momentum * v_h_bias + learn_rate * d_h_bias;
                v_v_bias = momentum * v_v_bias + learn_rate * d_v_bias; 
                v_v_sgma = momentum * v_v_sgma + learn_rate * learn_sgma * d_v_sgma;
                
                obj.weight      = obj.weight      + v_weight;
                obj.hidden_bias = obj.hidden_bias + v_h_bias;
                obj.visual_bias = obj.visual_bias + v_v_bias;
                obj.visual_sgma = obj.visual_sgma + v_v_sgma;
                obj.visual_sgma(obj.visual_sgma <= 5e-3) = 5e-3;
            end
        end
        
        function y = reconstruct(obj,x)
            N = size(x,2); % 样本的个数
            h_bias = repmat(obj.hidden_bias,1,N);
            v_bias = repmat(obj.visual_bias,1,N);
            v_sgma = repmat(obj.visual_sgma,1,N);
            
            h_field_0 = learn.sigmoid(obj.weight * (x ./ v_sgma) + h_bias);
            h_state_0 = learn.sample(h_field_0);
            y = v_sgma .* (obj.weight'* h_state_0) + v_bias;
        end
        
    end
    
    methods (Access = private)
        function [d_weight,d_h_bias,d_v_bias,d_v_sgma,r_error] = CD1(obj, minibatch)
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
            v_sgma = repmat(obj.visual_sgma,1,N);
            
            h_field_0 = learn.sigmoid(obj.weight * (minibatch ./ v_sgma) + h_bias);
            h_state_0 = learn.sample(h_field_0);
            v_field_1 = v_sgma .* (obj.weight'* h_state_0) + v_bias;
            v_state_1 = v_field_1 + v_sgma .* randn(size(v_field_1));
            h_field_1 = learn.sigmoid(obj.weight * (v_state_1 ./ v_sgma) + h_bias);
            h_state_1 = learn.sample(h_field_1);
            
            r_error =  sum(sum((v_field_1 - minibatch).^2)) / N; %计算在整个minibatch上的平均重建误差
            
            d_weight = (h_field_0 * (minibatch ./ v_sgma)' - h_field_1 * (v_state_1 ./ v_sgma)') / N;
            d_h_bias = (h_field_0 - h_field_1) * ones(N,1) / N;
            d_v_bias = ((minibatch - v_state_1) ./ (v_sgma.^2)) * ones(N,1) / N;
            
            d_v_sgma1 = (((minibatch - v_bias).^2) ./ (v_sgma.^3)) * ones(N,1) / N;
            d_v_sgma2 = ((h_state_0 * (minibatch ./ (v_sgma.^2))') .* obj.weight)' * ones(obj.num_hidden,1) / N;
            d_v_sgma3 = (((v_state_1 - v_bias).^2) ./ (v_sgma.^3)) * ones(N,1) / N;
            d_v_sgma4 = ((h_state_1 * (v_state_1 ./ (v_sgma.^2))') .* obj.weight)' * ones(obj.num_hidden,1) / N;
            d_v_sgma = (d_v_sgma1 - d_v_sgma2) - (d_v_sgma3 - d_v_sgma4);
        end
        
        function obj = initialize_weight(obj,train_data)
            %INTIALIZE 初始化权值矩阵为0附近的小随机数，初始化显层神经元的偏置为先验概率，初始化隐层神经元的偏置为0.
            obj.weight = 0.01 * randn(size(obj.weight));
            obj.visual_bias = mean(train_data,2);
            obj.hidden_bias = zeros(size(obj.hidden_bias));
            obj.visual_sgma = ones(size(obj.visual_sgma));
        end
    end
    
    methods(Static)
        function [gbrbm,e] = unit_test()
            clear all;
            close all;
            rng(1);
                
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
            
            data = trwhitening * Z;
            data = reshape(data,D,S,M); 
            
            gbrbm = learn.GBRBM(D,100);
            gbrbm = gbrbm.initialize(data);
            gbrbm = gbrbm.pretrain(data,1e-6,1e-3,1e-4,1e6);
            
            recon_data = dewhitening * gbrbm.reconstruct(trwhitening * Z) + AVE_X;
            
            e = 1;
        end
        
         function [gbrbm,e] = unit_test2()
            clear all;
            close all;
            rng(1);
            
            data = [1:10; 10:-1:1]'; D = 10; S = 2; M = 500;
            data = repmat(data,1,1,500); N = S * M;
            
            gbrbm = learn.GBRBM(D,60);
            gbrbm = gbrbm.initialize(data);
            gbrbm = gbrbm.pretrain(data,[1e-6,1e-2],1e-2,1e6);
            
            data = reshape(data,D,[]);
            recon_data = gbrbm.reconstruct(data);
            e = sum(sum((recon_data - data).^2)) / N;
         end
        
         function [gbrbm,e] = unit_test3()
            clear all;
            close all;
            rng(1);
            
            [data,label,test_data,test_label] = learn.import_mnist('./+learn/mnist.mat');
            [D,S,M] = size(data); data = data * 255; data = reshape(data,D,[]);
     
            data = reshape(data,D,S,M);
            
            gbrbm = learn.GBRBM(D,500);
            gbrbm = gbrbm.initialize(data);
            gbrbm = gbrbm.pretrain(data,[1e-8,1e-4],1e-2,1e6);
            
            save('gbrbm.mat','gbrbm');
         
            data = reshape(data,D,[]);
            recon_data = gbrbm.reconstruct(data);
            e = sum(sum((recon_data - data).^2)) / (S*M);
        end
    end
    
end


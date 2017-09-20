classdef SAE < learn.neural.StackedRBM
    %STACKED AUTO ENCODER 栈式自动编码器
    %   
    
    methods
        function obj = SAE(configure) % 构造函数
            obj@learn.neural.StackedRBM(configure); % 调用父类的构造函数
        end
    end
    
    methods
        function obj = train(obj,minibatchs,parameters)
            switch parameters.case
                case 1 % 全抽样的情况
                    obj = obj.train_case1(minibatchs,parameters);
                case 2 % 无抽样的情况
                    obj = obj.train_case2(minibatchs,parameters);
                otherwise
                    error('parameters.case的值错误！')
            end
        end
        
        function code = encode(obj,data,option)
            %ENCODE 给定数据，计算其编码 
            %
            data = obj.posterior(data);
            if strcmp(option,'sample')
                code = learn.sample(data{length(data)});
            elseif strcmp(option,'nosample')
                code = data{length(data)};
            elseif strcmp(option,'fix')
                code = data{length(data)} > 0.5;
            else
                error('option的值错误');
            end
        end
        
        function data = decode(obj,code)
            %DECODE 给定编码，计算其数据
            %   
            data = obj.likelihood(code);
            data = data{1};
        end
        
        function rebuild_data = rebuild(obj,data,option)
            %REBUILD 给定原数据，通过神经网络后计算其重建数据
            %      
            rebuild_data = obj.decode(obj.encode(data,option));
        end
    end
    
    methods(Access = private)
        function obj = train_case1(obj,minibatchs,parameters) 
            % train 训练函数
            % 使用UPDOWN算法进行训练，全抽样
            obj = obj.weightsync(); % 权值同步（解锁）
            
            [D,S,M] = size(minibatchs); L = obj.layer_num(); % D数据维度，S批的大小，M批的个数，L层的个数
            ob = learn.Observer('重建误差',1,M); %初始化观察者用来观察重建误差
                        
            recon_error_list = zeros(1,M);
            for m = 1:M
                minibatch = minibatchs(:,:,m);
                recon_minibatch = obj.rebuild(minibatch,'sample');
                recon_error_list(m) = sum(sum((recon_minibatch - minibatch).^2)) / S;
            end
            
            recon_error_ave_old = mean(recon_error_list); % 计算重建误差的均值
            ob = ob.initialize(recon_error_ave_old);      % 用均值初始化观察者
            
            learn_rate_min = min(parameters.learn_rate);
            learn_rate     = max(parameters.learn_rate);  % 将学习速度初始化为最大学习速度
            
            for it = 0:parameters.max_it
                m = mod(it,M) + 1;
                minibatch = minibatchs(:,:,m);
                
                pos_state = obj.posterior_sample(minibatch);
                neg_state = obj.likelihood_sample(pos_state{L+1}.state);
                neg_state{1}.state = neg_state{1}.proba;
                
                for l = L:-1:1
                    pre_pos_proba{l  } = obj.rbms{l}.likelihood(pos_state{l+1}.state);
                    pre_neg_proba{l+1} = obj.rbms{l}.posterior (neg_state{l  }.state);
                end
                
                % 更新产生权值
                for l = 1:L
                    obj.rbms{l}.weight_h2v = obj.rbms{l}.weight_h2v + learn_rate * ...
                        pos_state{l+1}.state * (pos_state{l}.state - pre_pos_proba{l})' / S;
                    obj.rbms{l}.visual_bias = obj.rbms{l}.visual_bias + learn_rate * ...
                        sum(pos_state{l}.state - pre_pos_proba{l},2) / S;
                end
                
                % 更新识别权值
                for l = 1:L
                    obj.rbms{l}.weight_v2h = obj.rbms{l}.weight_v2h + learn_rate * ...
                        (neg_state{l+1}.state - pre_neg_proba{l+1}) * neg_state{l}.state' / S;
                    obj.rbms{l}.hidden_bias = obj.rbms{l}.hidden_bias + learn_rate * ...
                        sum(neg_state{l+1}.state - pre_neg_proba{l+1},2) / S;
                end
                
                % 计算重建误差
                recon_minibatch = obj.rebuild(minibatch,'sample');
                recon_error = sum(sum((recon_minibatch - minibatch).^2)) / S;
                recon_error_list(m) = recon_error;
                recon_error_ave_new = mean(recon_error_list);
                
                if m == M
                    if recon_error_ave_new > recon_error_ave_old % 当经过M次迭代的平均重建误差不下降时
                        learn_rate = learn_rate / 2;         % 就缩减学习速度
                        if learn_rate < learn_rate_min       % 当学习速度小于最小学习速度时，退出
                            break;
                        end
                    end
                    recon_error_ave_old = recon_error_ave_new;
                end
                
                % 画图
                description = strcat('重建误差：',num2str(recon_error_ave_new));
                description = strcat(description,strcat('迭代次数:',num2str(it)));
                description = strcat(description,strcat('学习速度:',num2str(learn_rate)));
                % disp(description);
                ob = ob.showit(recon_error_ave_new,description);
            end
        end
        
        function obj = train_case2(obj,minibatchs,parameters) 
            % train_case2 训练函数
            % 使用UPDOWN算法进行训练，无抽样
            
            %% 权值同步（解锁）
            obj = obj.weightsync();
            
            %% D数据维度，S批的大小，M批的个数，L层的个数
            [D,S,M] = size(minibatchs); L = obj.layer_num(); 
                        
            recon_error_list = zeros(1,M);
            for m = 1:M
                minibatch = minibatchs(:,:,m);
                recon_minibatch = obj.rebuild(minibatch,'nosample');
                recon_error_list(m) = sum(sum((recon_minibatch - minibatch).^2)) / S;
            end
            
            recon_error_ave_old = mean(recon_error_list); % 计算重建误差的均值
            learn_rate_min = parameters.learn_rate_min;
            learn_rate     = parameters.learn_rate_max;   % 将学习速度初始化为最大学习速度
            
            %% 初始化动量值
            inc = cell(1,L);
            for l = 1:L
                inc{l}.weight_v2h  = zeros(size(obj.rbms{l}.weight_v2h));
                inc{l}.weight_h2v  = zeros(size(obj.rbms{l}.weight_h2v));
                inc{l}.visual_bias = zeros(size(obj.rbms{l}.visual_bias));
                inc{l}.hidden_bias = zeros(size(obj.rbms{l}.hidden_bias));
            end
            
            %% 开始迭代
            for it = 0:parameters.max_it
                m = mod(it,M) + 1;
                minibatch = minibatchs(:,:,m);
                
                pos_proba = obj.posterior(minibatch);
                neg_proba = obj.likelihood(pos_proba{L+1});
                
                for l = L:-1:1
                    pre_pos_proba{l  } = obj.rbms{l}.likelihood(pos_proba{l+1});
                    pre_neg_proba{l+1} = obj.rbms{l}.posterior (neg_proba{l  });
                end
                
                
                for l = 1:L
                    inc{l}.weight_h2v  = parameters.momentum * inc{l}.weight_h2v  + (1 - parameters.momentum) * learn_rate * pos_proba{l+1} * (pos_proba{l} - pre_pos_proba{l})' / S;
                    inc{l}.visual_bias = parameters.momentum * inc{l}.visual_bias + (1 - parameters.momentum) * learn_rate * sum(pos_proba{l} - pre_pos_proba{l},2) / S;
                    inc{l}.weight_v2h  = parameters.momentum * inc{l}.weight_v2h  + (1 - parameters.momentum) * learn_rate * (neg_proba{l+1} - pre_neg_proba{l+1}) * neg_proba{l}' / S;
                    inc{l}.hidden_bias = parameters.momentum * inc{l}.hidden_bias + (1 - parameters.momentum) * learn_rate * sum(neg_proba{l+1} - pre_neg_proba{l+1},2) / S;
                end
                
                %% 更新产生权值
                for l = 1:L
                    obj.rbms{l}.weight_h2v  = obj.rbms{l}.weight_h2v  + inc{l}.weight_h2v;
                    obj.rbms{l}.visual_bias = obj.rbms{l}.visual_bias + inc{l}.visual_bias;
                end
                
                %% 更新识别权值
                for l = 1:L
                    obj.rbms{l}.weight_v2h  = obj.rbms{l}.weight_v2h  + inc{l}.weight_v2h;
                    obj.rbms{l}.hidden_bias = obj.rbms{l}.hidden_bias + inc{l}.hidden_bias;
                end
                
                %% 计算重建误差
                recon_minibatch = obj.rebuild(minibatch,'nosample');
                recon_error = sum(sum((recon_minibatch - minibatch).^2)) / S;
                recon_error_list(m) = recon_error;
                recon_error_ave_new = mean(recon_error_list);
                
                if m == M
                    if recon_error_ave_new > recon_error_ave_old % 当经过M次迭代的平均重建误差不下降时
                        learn_rate = learn_rate / 10;            % 就缩减学习速度
                        if learn_rate < learn_rate_min           % 当学习速度小于最小学习速度时，退出
                            break;
                        end
                    end
                    recon_error_ave_old = recon_error_ave_new;
                end
                
                %% 画图
                disp(sprintf('重建误差:%f 迭代次数:%d 学习速度:%f',recon_error_ave_new,it,learn_rate));
            end
        end
    end
    
    methods(Static)
        function [] = unit_test()
            clear all;
            close all;
            rng(1);
            
            [data,~,~,~] = learn.data.import_mnist('./+learn/+data/mnist.mat');
            [D,S,M] = size(data); N = S * M;
            
            configure = [D,500,500,64];
            sae = learn.neural.SAE(configure);
            
            parameters.learn_rate = 1e-1;
            parameters.max_it = M*100;
            parameters.decay = 9;
            sae = sae.pretrain(data,parameters);
            save('sae_mnist_pretrain.mat','sae');
            % load('sae_mnist_pretrain.mat');
            
            data = reshape(data,D,[]);
            recon_data = sae.rebuild(data,'nosample');
            error = sum(sum((recon_data - data).^2)) / N;
            disp(sprintf('pretrain-error:%f',error));
            
            data = reshape(data,D,S,M);
            clear parameters;
            parameters.learn_rate_max = 1e-1;
            parameters.learn_rate_min = 1e-6;
            parameters.momentum = 0.9;
            parameters.max_it = 1e6;
            parameters.case = 2; % 无抽样的情况
            sae = sae.train(data,parameters);
            save('sae_mnist_finetune.mat','sae');
            
            data = reshape(data,D,[]);
            recon_data = sae.rebuild(data,'nosample');
            error = sum(sum((recon_data - data).^2)) / N;
            disp(sprintf('finetune-error:%f',error));
        end
    end
end


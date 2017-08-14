classdef SAE
    %STACKED AUTO ENCODER 栈式自动编码器
    %   
    
    properties
        encoder;
        decoder;
    end
    
    methods
        function obj = SAE(configure) 
            % 构造函数
            obj.encoder = learn.StackedRBM(configure);
            obj.decoder = obj.encoder;
        end
    end
    
    methods
        function obj = pretrain(obj,minibatchs,parameters) 
            %pretrain 使用CD1快速算法，逐层训练
            obj.encoder = obj.encoder.pretrain(minibatchs,parameters); % 对encoder进行预训练
            obj.decoder = obj.encoder; % 将decoder初始化为与encoder相同（权值解锁）
        end
        
        function obj = train(obj,minibatchs,parameters) 
            % train 训练函数
            % 使用wake-sleep算法进行训练
            
            [D,S,M] = size(minibatchs);  % D数据维度，S是minibatch的大小，M是minibatch的个数
            ob = learn.Observer('重建误差',1,M); %初始化观察者用来观察重建误差
                        
            recon_error_list = zeros(1,M);
            for m = 1:M
                minibatch = minibatchs(:,:,m);
                recon_minibatch = obj.rebuild(minibatch);
                recon_error_list(m) = sum(sum((recon_minibatch - minibatch).^2)) / S;
            end
            
            recon_error_ave_old = mean(recon_error_list); % 计算重建误差的均值
            ob = ob.initialize(recon_error_ave_old);      % 用均值初始化观察者
            
            L = obj.decoder.layer_num();
            
            learn_rate_min = min(parameters.learn_rate);
            learn_rate     = max(parameters.learn_rate);  % 将学习速度初始化为最大学习速度
            
            for it = 0:parameters.max_it
                m = mod(it,M) + 1;
                minibatch = minibatchs(:,:,m);
                
                delta = obj.positive_phase(minibatch); % wake
                for l = 1:L
                    obj.decoder.rbms{l}.weight      = obj.decoder.rbms{l}.weight      + learn_rate * delta{l}.weight;
                    obj.decoder.rbms{l}.visual_bias = obj.decoder.rbms{l}.visual_bias + learn_rate * delta{l}.bias;
                end
                
                clear delta;
                
                delta = obj.negative_phase(minibatch); % sleep
                for l = 1:L
                    obj.encoder.rbms{l}.weight      = obj.encoder.rbms{l}.weight      + learn_rate * delta{l}.weight;
                    obj.encoder.rbms{l}.hidden_bias = obj.encoder.rbms{l}.hidden_bias + learn_rate * delta{l}.bias;
                end
                
                % 计算重建误差
                recon_minibatch = obj.rebuild(minibatch);
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
        
        function code = encode(obj,data)
            %ENCODE 给定数据，计算其编码 
            %
            data = obj.encoder.posterior(data);
            code = learn.sample(data);
        end
        
        function data = decode(obj,code)
            %DECODE 给定编码，计算其数据
            %   
            data = obj.decoder.likelihood(code);
        end
        
        function rebuild_data = rebuild(obj,data)
            %REBUILD 给定原数据，通过神经网络后计算其重建数据
            %      
            rebuild_data = obj.decode(obj.encode(data));
        end
        
        function delta = positive_phase(obj,minibatch)
            %positive_phase wake阶段
            %   wake训练阶段，encoder的参数不变，调整decoder的参数
            
            [D,S] = size(minibatch); % D数据维度，S样本个数
            L = obj.decoder.layer_num(); % L解码器层数
            
            % 对编码器进行抽样
            state_encoder = obj.encoder.posterior_sample(minibatch); 
            
            % 对解码器进行抽样
            state_decoder{L+1}.state = state_encoder{L+1}.state;
            for l = L:-1:1  
                state_decoder{l}.proba = obj.decoder.rbms{l}.likelihood(state_decoder{l+1}.state);
                state_decoder{l}.state = state_encoder{l}.state;
            end
            
            for l = L:-1:1
                x = state_decoder{l}.state;
                y = state_decoder{l}.proba;
                z = state_decoder{l+1}.state;
                
                delta{l}.bias   = sum(x - y,2) / S;
                delta{l}.weight = z * (x - y)' / S;
            end
        end
        
        function delta = negative_phase(obj,minibatch)
            %negative_phase sleep阶段
            % sleep训练阶段，decoder的参数不变，调整encoder的参数
            
            [D,S] = size(minibatch); % D数据维度，S样本个数
            L = obj.encoder.layer_num(); % L编码器层数
            
            % 先计算minibatch的码字
            code = obj.encoder.posterior_sample(minibatch); 
            code = code{L+1}.state;
            
            % 对解码器进行抽样
            state_decoder = obj.decoder.likelihood_sample(code); 
            
            % 对编码器进行抽样
            state_encoder = cell(1,L+1);
            state_encoder{1}.state = state_decoder{1}.proba;
            for l = 2:(L+1)
                state_encoder{l}.proba = obj.encoder.rbms{l-1}.posterior(state_encoder{l-1}.state);
                state_encoder{l}.state = state_decoder{l}.state;
            end
            
            delta = cell(1,L);
            for l = 1:L
                x = state_encoder{l+1}.state;
                y = state_encoder{l+1}.proba;
                z = state_encoder{l}.state;
                
                delta{l}.bias   = sum(x - y,2)    / S;
                delta{l}.weight = (z * (x - y)')' / S;
            end
        end
    end
    
    methods(Static)
        function [sae,e] = unit_test()
            clear all;
            close all;
            [data,~,~,~] = learn.import_mnist('./+learn/mnist.mat');
            [D,S,M] = size(data); N = S * M;
     
            configure = [D,500,500,2000,256];
            sae = learn.SAE(configure);
            
            parameters.learn_rate = [1e-6,1e-2];
            parameters.weight_cost = 1e-4;
            parameters.max_it = 1e6;
            sae = sae.pretrain(data,parameters);
            
            save('sae_mnist.mat','sae');
         
            data = reshape(data,D,[]);
            recon_data = sae.rebuild(data);
            e = sum(sum((255*recon_data - 255*data).^2)) / N;
        end
    end
end


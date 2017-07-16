classdef StackedAutoEncoder
    %AUTOSTACKENCODER 栈式自动编码器
    %   
    
    properties
        encoder;
        decoder;
    end
    
    methods
        function obj = StackedAutoEncoder(configure) 
            %AutoStackEncoder 构造函数
            obj.encoder = ML.StackedRestrictedBoltzmannMachine(configure);
            obj.decoder = obj.encoder;
        end
    end
    
    methods
        function obj = pretrain(obj,minibatchs,learn_rate_min,learn_rate_max,max_it) 
            %pretrain 使用CD1快速算法，逐层训练
            obj.encoder = obj.encoder.pretrain(minibatchs,learn_rate_min,learn_rate_max,max_it); % 对encoder进行预训练
            obj.decoder = obj.encoder; % 将decoder初始化为与encoder相同（权值解锁）
        end
        
        function obj = train(obj,minibatchs,learn_rate_min,learn_rate_max,max_it) 
            % train 训练函数
            % 使用wake-sleep算法进行训练
            
            minibatch_num  = length(minibatchs);  % minibatch的个数
            ob_window_size = minibatch_num;       % 观察窗口的大小设置为minibatch的个数
            ob_var_num = 1; %跟踪变量的个数
            ob = ML.Observer('重建误差',ob_var_num,ob_window_size,'xxx'); %初始化观察者用来观察重建误差
            
            r_error_list = zeros(1,ob_window_size);
            for minibatch_idx = 1:minibatch_num
                minibatch = minibatchs{minibatch_idx};
                r_minibatch = obj.rebuild(minibatch);
                r_error_list(minibatch_idx) = sum(sum(abs(r_minibatch - minibatch))) / size(minibatch,2);
            end
            
            r_error_ave_old = mean(r_error_list); % 计算重建误差的均值
            ob = ob.initialize(r_error_ave_old);   % 用均值初始化观察者
            
            layers_num_d = obj.decoder.layer_num();
            layers_num_e = obj.encoder.layer_num();
            
            learn_rate = learn_rate_max;          % 将学习速度初始化为最大学习速度
            
            for it = 0:max_it
                minibatch_idx = mod(it,minibatch_num) + 1;
                minibatch = minibatchs{minibatch_idx};
                
                delta = obj.wake(minibatch); % wake 阶段
                
                obj.decoder.rbm_layers{layers_num_d}.hidden_bias = obj.decoder.rbm_layers{layers_num_d}.hidden_bias + ...
                    learn_rate * delta{layers_num_d+1}.v_bias;
                
                for n = 1:layers_num_d
                    obj.decoder.rbm_layers{n}.weight = obj.decoder.rbm_layers{n}.weight + learn_rate * delta{n}.weight;
                    obj.decoder.rbm_layers{n}.visual_bias = obj.decoder.rbm_layers{n}.visual_bias + learn_rate * delta{n}.v_bias;
                end
                
                delta_sleep = obj.sleep_sample(minibatch); % sleep 阶段
                for n = 1:num_of_layers
                    velocity_sleep(n).weight = momentum * velocity_sleep(n).weight + learn_rate * delta_sleep(n).weight;
                    obj.encoder_layers(n).rbm.weight = obj.encoder_layers(n).rbm.weight + velocity_sleep(n).weight;
                    velocity_sleep(n).hidden_bias = momentum * velocity_sleep(n).hidden_bias + learn_rate * delta_sleep(n).hidden_bias;
                    obj.encoder_layers(n).rbm.hidden_bias = obj.encoder_layers(n).rbm.hidden_bias + velocity_sleep(n).hidden_bias;
                end
                
                % 计算重建误差
                r_minibatch = obj.rebuild(minibatch);
                rebuild_error = sum(sum(abs(r_minibatch - minibatch))) / size(minibatch,2);
                rebuild_error_list(mod(it,ob_window_size)+1) = rebuild_error;
                rebuild_error_average = mean(rebuild_error_list);
                
                if mod(it,ob_window_size) == 0
                    if rebuild_error_average > rebuild_error_average_old % 当经过N次迭代的平均重建误差不下降时
                        learn_rate = learn_rate / 2; % 就缩减学习速度
                        if learn_rate < learn_rate_min % 当学习速度小于最小学习速度时，退出
                            break;
                        end
                    end
                    rebuild_error_average_old = rebuild_error_average;
                end
                
                % 画图
                titlename = strcat(strcat(strcat('wake learning - step : ',num2str(it)),'/ '),num2str(max_it));
                titlename = strcat(titlename,strcat(';  rate : ',num2str(learn_rate)));
                ob = ob.showit(rebuild_error_average,titlename);
            end
        end
        
        function code = encode(obj,data)
            %ENCODE 给定数据，计算其编码 
            %
            data = obj.encoder.posterior(data);
            code = ML.sample(data);
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
        
        function delta = wake(obj,minibatch)
            %WAKE wake阶段
            %   wake训练阶段，encoder的参数不变，调整decoder的参数
            
            N = size(minibatch,2); % 数据样本个数
            layer_num_e = obj.encoder.layer_num();
            layer_num_d = obj.decoder.layer_num();
            
            [~,state_encoder] = obj.encoder.posterior_sample(minibatch); % 对编码器进行抽样
            [~,state_decoder] = obj.decoder.likelihood_sample(state_encoder{layer_num_e}.h_state); % 对解码器进行抽样
            state_decoder{layer_num_d}.h_field = repmat(ML.sigmoid(obj.decoder.rbm_layers{layer_num_d}.hidden_bias),1,N);
            state_decoder{1}.v_state = minibatch;
            
            x = state_decoder{layer_num_d}.h_state;
            y = state_decoder{layer_num_d}.h_field;
            delta{layer_num_d+1}.v_bias = sum(x - y,2) / N;
            
            for n = layer_num_d:-1:1
                x = state_decoder{n}.h_state;
                a = state_decoder{n}.v_state;
                b = state_decoder{n}.v_field;
                delta{n}.v_bias = sum(a - b,2) / N;
                delta{n}.weight = x * (a - b)' / N;
            end
        end
        
        function delta = sleep(obj,minibatch)
            %SLEEP sleep阶段
            % sleep训练阶段，decoder的参数不变，调整encoder的参数
            、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、
            N = size(minibatch,2); % 数据样本个数
            layer_num_e = obj.encoder.layer_num();
            layer_num_d = obj.decoder.layer_num();
            
            decoder(num_of_layers).hidden_s = obj.encode_sample(minibatch);
            
            for n = num_of_layers:-1:1
                decoder(n).visual_p = obj.decoder_layers(n).rbm.likelihood(decoder(n).hidden_s);
                if n > 1
                    decoder(n).visual_s = DML.sample(decoder(n).visual_p);
                else
                    decoder(n).visual_s = decoder(n).visual_p;
                end
                if n > 1
                    decoder(n-1).hidden_s = decoder(n).visual_s;
                end
            end
            
            encoder(1).visual_s = decoder(1).visual_s;
            for n = 1:num_of_layers
                encoder(n).hidden_p = obj.encoder_layers(n).rbm.posterior(encoder(n).visual_s);
                encoder(n).hidden_s = decoder(n).hidden_s;
                if n < num_of_layers
                    encoder(n+1).visual_s = encoder(n).hidden_s;
                end
            end
            
            for n = 1:num_of_layers
                x = encoder(n).hidden_s;
                y = encoder(n).hidden_p;
                z = encoder(n).visual_s;
                delta(n).hidden_bias = sum(x - y,2) / size(minibatch,2);
                delta(n).weight =  (z * (x - y)')' / size(minibatch,2);
            end
        end
    end
end


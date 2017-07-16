classdef StackedAutoEncoder
    %AUTOSTACKENCODER ջʽ�Զ�������
    %   
    
    properties
        encoder;
        decoder;
    end
    
    methods
        function obj = StackedAutoEncoder(configure) 
            %AutoStackEncoder ���캯��
            obj.encoder = ML.StackedRestrictedBoltzmannMachine(configure);
            obj.decoder = obj.encoder;
        end
    end
    
    methods
        function obj = pretrain(obj,minibatchs,learn_rate_min,learn_rate_max,max_it) 
            %pretrain ʹ��CD1�����㷨�����ѵ��
            obj.encoder = obj.encoder.pretrain(minibatchs,learn_rate_min,learn_rate_max,max_it); % ��encoder����Ԥѵ��
            obj.decoder = obj.encoder; % ��decoder��ʼ��Ϊ��encoder��ͬ��Ȩֵ������
        end
        
        function obj = train(obj,minibatchs,learn_rate_min,learn_rate_max,max_it) 
            % train ѵ������
            % ʹ��wake-sleep�㷨����ѵ��
            
            minibatch_num  = length(minibatchs);  % minibatch�ĸ���
            ob_window_size = minibatch_num;       % �۲촰�ڵĴ�С����Ϊminibatch�ĸ���
            ob_var_num = 1; %���ٱ����ĸ���
            ob = ML.Observer('�ؽ����',ob_var_num,ob_window_size,'xxx'); %��ʼ���۲��������۲��ؽ����
            
            r_error_list = zeros(1,ob_window_size);
            for minibatch_idx = 1:minibatch_num
                minibatch = minibatchs{minibatch_idx};
                r_minibatch = obj.rebuild(minibatch);
                r_error_list(minibatch_idx) = sum(sum(abs(r_minibatch - minibatch))) / size(minibatch,2);
            end
            
            r_error_ave_old = mean(r_error_list); % �����ؽ����ľ�ֵ
            ob = ob.initialize(r_error_ave_old);   % �þ�ֵ��ʼ���۲���
            
            layers_num_d = obj.decoder.layer_num();
            layers_num_e = obj.encoder.layer_num();
            
            learn_rate = learn_rate_max;          % ��ѧϰ�ٶȳ�ʼ��Ϊ���ѧϰ�ٶ�
            
            for it = 0:max_it
                minibatch_idx = mod(it,minibatch_num) + 1;
                minibatch = minibatchs{minibatch_idx};
                
                delta = obj.wake(minibatch); % wake �׶�
                
                obj.decoder.rbm_layers{layers_num_d}.hidden_bias = obj.decoder.rbm_layers{layers_num_d}.hidden_bias + ...
                    learn_rate * delta{layers_num_d+1}.v_bias;
                
                for n = 1:layers_num_d
                    obj.decoder.rbm_layers{n}.weight = obj.decoder.rbm_layers{n}.weight + learn_rate * delta{n}.weight;
                    obj.decoder.rbm_layers{n}.visual_bias = obj.decoder.rbm_layers{n}.visual_bias + learn_rate * delta{n}.v_bias;
                end
                
                delta_sleep = obj.sleep_sample(minibatch); % sleep �׶�
                for n = 1:num_of_layers
                    velocity_sleep(n).weight = momentum * velocity_sleep(n).weight + learn_rate * delta_sleep(n).weight;
                    obj.encoder_layers(n).rbm.weight = obj.encoder_layers(n).rbm.weight + velocity_sleep(n).weight;
                    velocity_sleep(n).hidden_bias = momentum * velocity_sleep(n).hidden_bias + learn_rate * delta_sleep(n).hidden_bias;
                    obj.encoder_layers(n).rbm.hidden_bias = obj.encoder_layers(n).rbm.hidden_bias + velocity_sleep(n).hidden_bias;
                end
                
                % �����ؽ����
                r_minibatch = obj.rebuild(minibatch);
                rebuild_error = sum(sum(abs(r_minibatch - minibatch))) / size(minibatch,2);
                rebuild_error_list(mod(it,ob_window_size)+1) = rebuild_error;
                rebuild_error_average = mean(rebuild_error_list);
                
                if mod(it,ob_window_size) == 0
                    if rebuild_error_average > rebuild_error_average_old % ������N�ε�����ƽ���ؽ����½�ʱ
                        learn_rate = learn_rate / 2; % ������ѧϰ�ٶ�
                        if learn_rate < learn_rate_min % ��ѧϰ�ٶ�С����Сѧϰ�ٶ�ʱ���˳�
                            break;
                        end
                    end
                    rebuild_error_average_old = rebuild_error_average;
                end
                
                % ��ͼ
                titlename = strcat(strcat(strcat('wake learning - step : ',num2str(it)),'/ '),num2str(max_it));
                titlename = strcat(titlename,strcat(';  rate : ',num2str(learn_rate)));
                ob = ob.showit(rebuild_error_average,titlename);
            end
        end
        
        function code = encode(obj,data)
            %ENCODE �������ݣ���������� 
            %
            data = obj.encoder.posterior(data);
            code = ML.sample(data);
        end
        
        function data = decode(obj,code)
            %DECODE �������룬����������
            %   
            data = obj.decoder.likelihood(code);
        end
        
        function rebuild_data = rebuild(obj,data)
            %REBUILD ����ԭ���ݣ�ͨ���������������ؽ�����
            %      
            rebuild_data = obj.decode(obj.encode(data));
        end
        
        function delta = wake(obj,minibatch)
            %WAKE wake�׶�
            %   wakeѵ���׶Σ�encoder�Ĳ������䣬����decoder�Ĳ���
            
            N = size(minibatch,2); % ������������
            layer_num_e = obj.encoder.layer_num();
            layer_num_d = obj.decoder.layer_num();
            
            [~,state_encoder] = obj.encoder.posterior_sample(minibatch); % �Ա��������г���
            [~,state_decoder] = obj.decoder.likelihood_sample(state_encoder{layer_num_e}.h_state); % �Խ��������г���
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
            %SLEEP sleep�׶�
            % sleepѵ���׶Σ�decoder�Ĳ������䣬����encoder�Ĳ���
            ����������������������������������������������������������������������
            N = size(minibatch,2); % ������������
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


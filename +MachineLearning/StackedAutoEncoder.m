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
            
            layers_num = obj.decoder.layer_num();
            
            learn_rate = learn_rate_max;          % ��ѧϰ�ٶȳ�ʼ��Ϊ���ѧϰ�ٶ�
            
            for it = 0:max_it
                minibatch_idx = mod(it,minibatch_num) + 1;
                minibatch = minibatchs{minibatch_idx};
                N = size(minibatch,2); 
                
                delta = obj.wake(minibatch); % wake �׶�
                
                obj.decoder.rbm_layers{layers_num}.hidden_bias = obj.decoder.rbm_layers{layers_num}.hidden_bias + ...
                    learn_rate * delta{layers_num}.h_bias;
                
                for n = 1:layers_num
                    obj.decoder.rbm_layers{n}.weight = obj.decoder.rbm_layers{n}.weight + learn_rate * delta{n}.weight;
                    obj.decoder.rbm_layers{n}.visual_bias = obj.decoder.rbm_layers{n}.visual_bias + learn_rate * delta{n}.v_bias;
                end
                
                clear delta;
                delta = obj.sleep(minibatch); % sleep �׶�
                
                for n = 1:layers_num
                    obj.encoder.rbm_layers{n}.weight = obj.encoder.rbm_layers{n}.weight + learn_rate * delta{n}.weight;
                    obj.encoder.rbm_layers{n}.hidden_bias = obj.encoder.rbm_layers{n}.hidden_bias + learn_rate * delta{n}.h_bias;
                end
                
                % �����ؽ����
                r_minibatch = obj.rebuild(minibatch);
                r_error = sum(sum(abs(r_minibatch - minibatch))) / N;
                r_error_list(mod(it,ob_window_size)+1) = r_error;
                r_error_ave_new = mean(r_error_list);
                
                if minibatch_idx == minibatch_num
                    if r_error_ave_new > r_error_ave_old % ������N�ε�����ƽ���ؽ����½�ʱ
                        learn_rate = learn_rate / 2; % ������ѧϰ�ٶ�
                        if learn_rate < learn_rate_min % ��ѧϰ�ٶ�С����Сѧϰ�ٶ�ʱ���˳�
                            break;
                        end
                    end
                    r_error_ave_old = r_error_ave_new;
                end
                
                % ��ͼ
                description = strcat(strcat(strcat('��������:',num2str(it)),'/ '),num2str(max_it));
                description = strcat(description,strcat('ѧϰ�ٶ�:',num2str(learn_rate)));
                ob = ob.showit(r_error_ave_new,description);
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
            layer_num = obj.decoder.layer_num();
            
            % �Ա��������г���
            [code,state_encoder] = obj.encoder.posterior_sample(minibatch); 
            
            % �Խ��������г���
            state_decoder{layer_num}.h_field = repmat(ML.sigmoid(obj.decoder.rbm_layers{layer_num}.hidden_bias),1,N);
            state_decoder{layer_num}.h_state = code;
            for n = layer_num:-1:1  
                state_decoder{n}.v_field = obj.decoder.rbm_layers{n}.likelihood(state_decoder{n}.h_state);
                state_decoder{n}.v_state = state_encoder{n}.v_state;
                if n > 1
                    state_decoder{n-1}.h_state = state_decoder{n}.v_state;
                    state_decoder{n-1}.h_field = state_decoder{n}.v_field;
                end
            end
            
            x = state_decoder{layer_num}.h_state;
            y = state_decoder{layer_num}.h_field;
            delta{layer_num}.h_bias = sum(x - y,2) / N;
            
            for n = layer_num:-1:1
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
            N = size(minibatch,2); % ������������
            layer_num = obj.encoder.layer_num();
            
            % �ȼ���minibatch������
            code = obj.encoder.posterior_sample(minibatch);
            
            % �Խ��������г���
            [~,state_decoder] = obj.decoder.likelihood_sample(code); 
            
            % �Ա��������г���
            state_encoder{1}.v_state = state_decoder{1}.v_field;
            for n = 1:layer_num
                state_encoder{n}.h_field = obj.encoder.rbm_layers{n}.posterior(state_encoder{n}.v_state);
                state_encoder{n}.h_state = state_decoder{n}.h_state;
                if n < layer_num
                    state_encoder{n+1}.v_state = state_encoder{n}.h_state;
                    state_encoder{n+1}.v_field = state_encoder{n}.h_field;
                end
            end
            
            for n = 1:layer_num
                x = state_encoder{n}.h_state;
                y = state_encoder{n}.h_field;
                z = state_encoder{n}.v_state;
                delta{n}.h_bias = sum(x - y,2) / N;
                delta{n}.weight =  (z * (x - y)')' / N;
            end
        end
    end
end


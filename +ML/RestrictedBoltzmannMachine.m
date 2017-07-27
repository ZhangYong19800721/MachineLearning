classdef RestrictedBoltzmannMachine
    %RESTRICTEDBOLTZMANNMACHINE Լ������������
    %   
    
    properties
        num_hidden; % ������Ԫ�ĸ���
        num_visual; % �ɼ���Ԫ�ĸ���
        weight;     % Ȩֵ����(num_hidden * num_visual)
        hidden_bias; % ������Ԫ��ƫ��
        visual_bias; % �ɼ���Ԫ��ƫ��
    end
    
    methods
        function obj = RestrictedBoltzmannMachine(num_visual,num_hidden) % ���캯��
            obj.num_hidden = num_hidden;
            obj.num_visual = num_visual;
            obj.weight = zeros(obj.num_hidden,obj.num_visual);
            obj.hidden_bias = zeros(obj.num_hidden,1);
            obj.visual_bias = zeros(obj.num_visual,1);
        end
    end
    
    methods
        function obj = construct(obj,weight,visual_bias,hidden_bias)
            %construct ʹ��Ȩֵ������Ԫƫ�á�����Ԫƫ��ֱ�ӹ���RBM
            [obj.num_hidden, obj.num_visual] = size(weight);
            obj.weight = weight;
            obj.hidden_bias = hidden_bias;
            obj.visual_bias = visual_bias;
        end
        
        function obj = initialize(obj,minibatchs) 
            %initialize ����ѵ�����ݣ��ɶ��minibatch��ɵ�ѵ�����ݼ��ϣ���ʼ��Ȩֵ��������Ԫƫ�ú�����Ԫƫ��
            %           minibatchs ��һ��Ԫ�����顣
           
            minibatch_num = length(minibatchs);
            minibatch_sum = zeros(size(minibatchs{1}));
            
            for n = 1:minibatch_num
                minibatch_sum = minibatch_sum + minibatchs{n};
            end
            
            minibatch_sum = minibatch_sum ./ minibatch_num;
            obj = obj.initialize_weight(minibatch_sum);
        end

        function obj = pretrain(obj,minibatchs,learn_rate_min,learn_rate_max,max_it) 
            %pretrain ��Ȩֵ����Ԥѵ��
            % ʹ��CD1�����㷨��Ȩֵ����Ԥѵ��
            
            minibatch_num = length(minibatchs); % �õ�minibatch�ĸ���
            ob_window_size = minibatch_num;     % �趨�۲촰�ڵĴ�СΪ
            ob_var_num = 1;                     % �趨�۲�����ĸ���
            ob = ML.Observer('�ؽ����',ob_var_num,ob_window_size,'xxx'); %��ʼ���۲��ߣ��۲��ؽ����
            
            % ��ʼ��velocity����
            v_weight = zeros(size(obj.weight));
            v_h_bias = zeros(size(obj.hidden_bias));
            v_v_bias = zeros(size(obj.visual_bias));
            
            % ��ʼ����������Ϊ0.5
            momentum = 0.5;
            
            r_error_list = zeros(1,ob_window_size);
            for idx = 1:minibatch_num  % ��ʼ���ؽ�����б���ƶ�ƽ��ֵ
                minibatch = minibatchs{idx};
                [~, ~, ~, r_error] = obj.CD1(minibatch);
                r_error_list(idx) = r_error;
            end
            r_error_ave_old = mean(r_error_list);
            ob = ob.initialize(r_error_ave_old);
            
            learn_rate = learn_rate_max; %��ʼ��ѧϰ�ٶ�
            
            for it = 0:max_it
                minibatch_idx = mod(it,minibatch_num)+1;  % ȡһ��minibatch
                minibatch = minibatchs{minibatch_idx};
                
                [d_weight, d_h_bias, d_v_bias, r_error] = obj.CD1(minibatch);
                r_error_list(minibatch_idx) = r_error;
                r_error_ave_new = mean(r_error_list);
                
                if minibatch_idx == minibatch_num % �����е�minibatch����Ѷ��һƪ��ʱ�򣨵���۲촰�����ұߵ�ʱ��
                    if r_error_ave_new > r_error_ave_old
                        learn_rate = learn_rate / 2;
                        if learn_rate < learn_rate_min
                            break;
                        end
                    end
                    r_error_ave_old = r_error_ave_new;
                end
                
                description = strcat(strcat(strcat('��������:',num2str(it)),'/'),num2str(max_it));
                description = strcat(description,strcat('ѧϰ�ٶ�:',num2str(learn_rate)));
                ob = ob.showit(r_error_ave_new,description);
                
                momentum = min([momentum * 1.01,0.9]); % �����������Ϊ0.9����ʼֵΪ0.5����Լ����60��֮�������ʴﵽ0.9��
                v_weight = momentum * v_weight + learn_rate * d_weight;
                v_h_bias = momentum * v_h_bias + learn_rate * d_h_bias;
                v_v_bias = momentum * v_v_bias + learn_rate * d_v_bias;
                
                obj.weight      = obj.weight      + v_weight;
                obj.hidden_bias = obj.hidden_bias + v_h_bias;
                obj.visual_bias = obj.visual_bias + v_v_bias;
            end
        end
        
        function h_state = posterior_sample(obj,v_state)
            % posterior_sample ���������ʲ���
            % �ڸ����Բ���Ԫȡֵ������£�������Ԫ���г���
            h_state = ML.sample(obj.posterior(v_state));
        end
        
        function h_field = posterior(obj,v_state) 
            %POSTERIOR ����������
            % �ڸ����Բ���Ԫȡֵ������£���������Ԫ�ļ������
            h_field = ML.sigmoid(obj.foreward(v_state));
        end
        
        function v_state = likelihood_sample(obj,h_state) 
            % likelihood_sample ������Ȼ���ʲ���
            % �ڸ���������Ԫȡֵ������£�������Ԫ���г���
            v_state = ML.sample(obj.likelihood(h_state));
        end
        
        function v_field = likelihood(obj,h_state) 
            % likelihood ������Ȼ����
            % �ڸ���������Ԫȡֵ������£���������Ԫ�ļ������
            v_field = ML.sigmoid(obj.backward(h_state));
        end
        
        function y = foreward(obj,x)
            y = obj.weight * x + repmat(obj.hidden_bias,1,size(x,2));
        end
        
        function x = backward(obj,y)
            x = obj.weight'* y + repmat(obj.visual_bias,1,size(y,2));
        end
    end
    
    methods (Access = private)
        function [d_weight,d_h_bias,d_v_bias,r_error] = CD1(obj, minibatch)
            % ʹ��Contrastive Divergence 1 (CD1)������Լ����������RBM�����п���ѵ��
            % ���룺
            %   minibatch��һ��ѵ�����ݣ�һ�д���һ��ѵ��������������ʾѵ�������ĸ���
            % ���
            %   d_weight, Ȩֵ����ĵ�����ֵ.
            %   d_h_bias, ������Ԫƫ��ֵ�ĵ�����ֵ.
            %   d_v_bias, �ɼ���Ԫƫ��ֵ�ĵ�����ֵ.
            %   r_error,  �ؽ����ֵ
            
            N = size(minibatch,2); % ѵ�������ĸ���
            h_bias = repmat(obj.hidden_bias,1,N);
            v_bias = repmat(obj.visual_bias,1,N);
            
            h_field_0 = ML.sigmoid(obj.weight * minibatch + h_bias);
            h_state_0 = ML.sample(h_field_0);
            v_field_1 = ML.sigmoid(obj.weight'* h_state_0 + v_bias);
            v_state_1 = ML.sample(v_field_1);
            h_field_1 = ML.sigmoid(obj.weight * v_state_1 + h_bias);
            
            r_error =  sum(sum(abs(v_field_1 - minibatch))) / N; %����������minibatch�ϵ�ƽ���ؽ����
            
            d_weight = (h_field_0 * minibatch' - h_field_1 * v_state_1') / N;
            d_h_bias = (h_state_0 - h_field_1) * ones(N,1) / N;
            d_v_bias = (minibatch - v_field_1) * ones(N,1) / N;
        end
        
        function obj = initialize_weight(obj,train_data)
            %INTIALIZE ��ʼ��Ȩֵ����Ϊ0������С���������ʼ���Բ���Ԫ��ƫ��Ϊ������ʣ���ʼ��������Ԫ��ƫ��Ϊ0.
            obj.weight = 0.01 * randn(size(obj.weight));
            x = sum(train_data,2) / size(train_data,2);
            x(x<=0) = x(x<=0) + 0.000001;
            x(x>=1) = x(x>=1) - 0.000001;
            obj.visual_bias = log(x./(1-x));
            obj.hidden_bias = zeros(size(obj.hidden_bias));
        end
    end
    
end


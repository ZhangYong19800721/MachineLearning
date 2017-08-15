classdef SoftmaxRBM
    %Softmax Restricted Boltzmann Machine ��Softmax��Ԫ��Լ������������
    %   
    
    properties
        num_hidden;  % ������Ԫ�ĸ���
        num_visual;  % �ɼ���Ԫ�ĸ���
        num_softmax; % softmax��Ԫ�ĸ���
        weight;      % Ȩֵ����(num_hidden * num_visual)
        hidden_bias; % ������Ԫ��ƫ��
        visual_bias; % �ɼ���Ԫ��ƫ��
    end
    
    methods
        function obj = SoftmaxRBM(num_softmax,num_visual,num_hidden) % ���캯��
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
            %construct ʹ��softmax��Ԫ������Ȩֵ������Ԫƫ�á�����Ԫƫ��ֱ�ӹ���RBM
            [obj.num_hidden, obj.num_visual] = size(weight);
            obj.num_softmax = num_softmax;
            obj.weight = weight;
            obj.hidden_bias = hidden_bias;
            obj.visual_bias = visual_bias;
        end
        
        function obj = initialize(obj,minibatchs,labels) 
            %initialize ����ѵ�����ݣ��ɶ��minibatch��ɵ�ѵ�����ݼ��ϣ���ʼ��Ȩֵ��������Ԫƫ�ú�����Ԫƫ��
            %           minibatchs ��һ��Ԫ�����顣
            
            [D,~,~] = size(minibatchs);
            [K,~,~] = size(labels);
            minibatchs = reshape(minibatchs,D,[]);
            labels     = reshape(labels    ,K,[]);
            obj = obj.initialize_weight(minibatchs,labels);
        end

        function obj = pretrain(obj,minibatchs,labels,parameters) 
            %pretrain ��Ȩֵ����Ԥѵ��
            % ʹ��CD1�����㷨��Ȩֵ����Ԥѵ��
            
            [D,S,M] = size(minibatchs); % �õ�minibatch�ĸ���
            ob = learn.Observer('�ؽ����',1,M); %��ʼ���۲��ߣ��۲��ؽ����
            
            % ��ʼ��velocity����
            v_weight      = zeros(size(obj.weight));
            v_h_bias = zeros(size(obj.hidden_bias));
            v_v_bias = zeros(size(obj.visual_bias));
            
            % ��ʼ����������Ϊ0.5
            momentum = 0.5;
            
            recon_error_list = zeros(1,M);
            for m = 1:M  % ��ʼ���ؽ�����б���ƶ�ƽ��ֵ
                label = labels(:,:,m);
                minibatch = minibatchs(:,:,m);
                [~, ~, ~, recon_error] = obj.CD1(minibatch,label);
                recon_error_list(m) = recon_error;
            end
            recon_error_ave_old = mean(recon_error_list);
            ob = ob.initialize(recon_error_ave_old);
            
            learn_rate_min = min(parameters.learn_rate);
            learn_rate     = max(parameters.learn_rate); %��ʼ��ѧϰ�ٶ�
            
            for it = 0:parameters.max_it
                m = mod(it,M)+1;  % ȡһ��minibatch
                label = labels(:,:,m);
                minibatch = minibatchs(:,:,m);
                
                [d_weight, d_h_bias, d_v_bias, recon_error] = obj.CD1(minibatch,label);
                recon_error_list(m) = recon_error;
                recon_error_ave_new = mean(recon_error_list);
                
                if m == M % �����е�minibatch����Ѷ��һƪ��ʱ�򣨵���۲촰�����ұߵ�ʱ��
                    if recon_error_ave_new > recon_error_ave_old
                        learn_rate = learn_rate / 2;
                        if learn_rate < learn_rate_min
                            break;
                        end
                    end
                    recon_error_ave_old = recon_error_ave_new;
                end
                
                description = strcat('�ؽ����:',num2str(recon_error_ave_new));
                description = strcat(description,strcat('��������:',num2str(it)));
                description = strcat(description,strcat('ѧϰ�ٶ�:',num2str(learn_rate)));
                disp(description);
                % ob = ob.showit(r_error_ave_new,description);
                
                momentum = min([momentum * 1.01,0.9]); % �����������Ϊ0.9����ʼֵΪ0.5����Լ����60��֮�������ʴﵽ0.9��
                v_weight = momentum * v_weight + learn_rate * (d_weight - parameters.weight_cost * obj.weight);
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
            h_state = learn.sample(obj.posterior(v_state));
        end
        
        function h_field = posterior(obj,v_state) 
            %POSTERIOR ����������
            % �ڸ����Բ���Ԫȡֵ������£���������Ԫ�ļ������
            N = size(v_state,2);
            h_field = learn.sigmoid(obj.weight * v_state + repmat(obj.hidden_bias,1,N));
        end
        
        function [s_state,v_state] = likelihood_sample(obj,h_state) 
            % likelihood_sample ������Ȼ���ʲ���
            % �ڸ���������Ԫȡֵ������£�������Ԫ���г���
            v_field = obj.likelihood(h_state);
            s_state = learn.sample_softmax(v_field(1:obj.num_softmax,:));
            v_state = learn.sample(v_field((obj.num_softmax+1):obj.num_visual,:));
        end
        
        function [s_field,v_field] = likelihood(obj,h_state) 
            % likelihood ������Ȼ����
            % �ڸ���������Ԫȡֵ������£���������Ԫ�ļ������
            N = size(h_state,2);
            v_sigma = obj.weight'* h_state + repmat(obj.visual_bias,1,N);
            s_field = learn.softmax(v_sigma(1:obj.num_softmax,:));
            v_field = learn.sigmoid(v_sigma((obj.num_softmax+1):obj.num_visual,:));
        end
        
        function y = classify(obj,x)
            %DISCRIMINATE �������ݵ㣬��������ݵķ���
            %
            N = size(x,2); % ����������ĸ���
            y = -1 * ones(1,N);
     
            for n = 1:N
                v_state = x(:,n);
                min_energy = inf;
                
                for class_idx = 1:obj.num_softmax
                    s_state = zeros(obj.num_softmax,1); s_state(class_idx) = 1;
                    % ���������Ԫ��Ӧ����������
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
            % ʹ��Contrastive Divergence 1 (CD1)������Լ����������RBM�����п���ѵ������RBM����Softmax��Ԫ��
            % ���룺
            %   minibatch��һ��ѵ�����ݣ�һ�д���һ��ѵ��������������ʾѵ�������ĸ���
            % ���:
            %   d_weight, Ȩֵ����ĵ�����ֵ.
            %   d_h_bias, ������Ԫƫ��ֵ�ĵ�����ֵ.
            %   d_v_bias, �ɼ���Ԫƫ��ֵ�ĵ�����ֵ.
            %   r_error,  �ؽ����ֵ
        
            N = size(minibatch,2); % ѵ�������ĸ���
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
            
            r_error =  sum(sum(([s_field_1;v_field_1] - [labels;minibatch]).^2)) / N; %����������train_data�ϵ�ƽ��reconstruction error
            
            d_weight = (h_field_0 * [labels;minibatch]' - h_field_1 * [s_state_1;v_state_1]') / N;
            d_h_bias = (h_state_0 - h_field_1) * ones(N,1) / N;
            d_v_bias = ([labels;minibatch] - [s_field_1;v_field_1]) * ones(N,1) / N;
        end
        
        function obj = initialize_weight(obj,train_data,train_label)
            %INTIALIZE ��ʼ��Ȩֵ����Ϊ0������С���������ʼ���Բ���Ԫ��ƫ��Ϊ������ʣ���ʼ��������Ԫ��ƫ��Ϊ0.
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


classdef SoftmaxRBM
    %Softmax Restricted Boltzmann Machine ��Softmax��Ԫ��Լ������������
    %   
    
    properties
        num_hidden;  % ������Ԫ�ĸ���
        num_visual;  % �ɼ���Ԫ�ĸ���
        num_softmax; % softmax��Ԫ�ĸ���
        weight_v2h;  % Ȩֵ����(num_hidden * num_visual)
        weight_s2h;  % Ȩֵ����(num_hidden * num_softmax)
        weight_h2v;  % Ȩֵ����(num_hidden * num_visual)
        weight_h2s;  % Ȩֵ����(num_hidden * num_softmax)
        hidden_bias; % ������Ԫ��ƫ��
        visual_bias; % �ɼ���Ԫ��ƫ��
        softmax_bias;% softmax��Ԫ��ƫ��
    end
    
    methods
        function obj = SoftmaxRBM(num_softmax,num_visual,num_hidden) % ���캯��
            obj.num_hidden = num_hidden;
            obj.num_visual = num_visual;
            obj.num_softmax = num_softmax;
            obj.weight_v2h = zeros(obj.num_hidden,obj.num_visual);
            obj.weight_s2h = zeros(obj.num_hidden,obj.num_softmax);
            obj.weight_h2v = [];
            obj.weight_h2s = [];
            obj.hidden_bias = zeros(obj.num_hidden,1);
            obj.visual_bias = zeros(obj.num_visual,1);
            obj.softmax_bias = zeros(obj.softmax_bias,1);
        end
    end
    
    methods
        function obj = construct(obj,weight_s2h,weight_v2h,weight_h2s,weight_h2v,softmax_bias,visual_bias,hidden_bias)
            %construct ʹ��softmax��Ԫ������Ȩֵ������Ԫƫ�á�����Ԫƫ��ֱ�ӹ���RBM
            obj.num_softmax = length(softmax_bias);
            obj.num_visual  = length(visual_bias);
            obj.num_hidden  = length(hidden_bias);
            
            obj.weight_s2h  = weight_s2h;
            obj.weight_v2h  = weight_v2h;
            obj.weight_h2s  = weight_h2s;
            obj.weight_h2v  = weight_h2v;
            
            obj.softmax_bias = softmax_bias;
            obj.hidden_bias  = hidden_bias;
            obj.visual_bias  = visual_bias;
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
            inc_w_s2h  = zeros(size(obj.weight_s2h));
            inc_w_v2h  = zeros(size(obj.weight_v2h));
            inc_h_bias = zeros(size(obj.hidden_bias));
            inc_s_bias = zeros(size(obj.softmax_bias));
            inc_v_bias = zeros(size(obj.visual_bias));
            
            % ��ʼ����������Ϊ0.5
            momentum = 0.5;
            
            recon_error_list = zeros(1,M);
            for m = 1:M  % ��ʼ���ؽ�����б���ƶ�ƽ��ֵ
                label = labels(:,:,m);
                minibatch = minibatchs(:,:,m);
                [~,~,~,~,~,recon_error] = obj.CD1(minibatch,label);
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
                
                [d_w_s2h,d_w_v2h,d_h_bias,d_s_bias,d_v_bias,recon_error] = obj.CD1(minibatch,label);
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
                % ob = ob.showit(recon_error_ave_new,description);
                
                momentum = min([momentum * 1.01,0.9]); % �����������Ϊ0.9����ʼֵΪ0.5����Լ����60��֮�������ʴﵽ0.9��
                inc_w_s2h  = momentum * inc_w_s2h  + learn_rate * (d_w_s2h - parameters.weight_cost * obj.weight_s2h);
                inc_w_v2h  = momentum * inc_w_v2h  + learn_rate * (d_w_v2h - parameters.weight_cost * obj.weight_v2h);
                inc_h_bias = momentum * inc_h_bias + learn_rate * d_h_bias;
                inc_s_bias = momentum * inc_s_bias + learn_rate * d_s_bias;
                inc_v_bias = momentum * inc_v_bias + learn_rate * d_v_bias;
                
                obj.weight_s2h   = obj.weight_s2h   + inc_w_s2h;
                obj.weight_v2h   = obj.weight_v2h   + inc_w_v2h;
                obj.hidden_bias  = obj.hidden_bias  + inc_h_bias;
                obj.softmax_bias = obj.softmax_bias + inc_s_bias;
                obj.visual_bias  = obj.visual_bias  + inc_v_bias;
            end
        end
        
        function h_state = posterior_sample(obj,s_state,v_state)
            % posterior_sample ���������ʲ���
            % �ڸ����Բ���Ԫȡֵ������£�������Ԫ���г���
            h_state = learn.sample(obj.posterior(s_state,v_state));
        end
        
        function h_field = posterior(obj,s_state,v_state) 
            %POSTERIOR ����������
            % �ڸ����Բ���Ԫȡֵ������£���������Ԫ�ļ������
            h_field = learn.sigmoid(obj.foreward(s_state,v_state));
        end
        
        function [s_state,v_state] = likelihood_sample(obj,h_state) 
            % likelihood_sample ������Ȼ���ʲ���
            % �ڸ���������Ԫȡֵ������£�������Ԫ���г���
            [s_field,v_field] = obj.likelihood(h_state);
            s_state = learn.sample_softmax(s_field);
            v_state = learn.sample(v_field);
        end
        
        function [s_field,v_field] = likelihood(obj,h_state) 
            % likelihood ������Ȼ����
            % �ڸ���������Ԫȡֵ������£���������Ԫ�ļ������
            [s,x] = obj.backward(h_state);
            s_field = learn.softmax(s);
            v_field = learn.sigmoid(x);
        end
        
        function y = foreward(obj,s,x)
            y = [obj.weight_s2h obj.weight_v2h] * [s;x] + repmat(obj.hidden_bias,1,size(x,2));
        end
        
        function [s,x] = backward(obj,y)
            if isempty(obj.weight_h2v)
                s = obj.weight_s2h'* y + repmat(obj.softmax_bias,1,size(y,2));
                x = obj.weight_v2h'* y + repmat(obj.visual_bias, 1,size(y,2));
            else
                s = obj.weight_h2s'* y + repmat(obj.softmax_bias,1,size(y,2));
                x = obj.weight_h2v'* y + repmat(obj.visual_bias, 1,size(y,2));
            end
        end
        
        function [c,y] = rebuild(obj,s,x)
            z = obj.posterior_sample(s,x);
            [c,y] = obj.likelihood(z);
        end
        
        function obj = weightsync(obj)
            obj.weight_h2s = obj.weight_s2h;
            obj.weight_h2v = obj.weight_v2h;
        end
        
        function y = classify(obj,x)
            %classify �������ݵ㣬��������ݵķ���
            %
            N = size(x,2); % ����������ĸ���
            E = inf * ones(obj.num_softmax,N);
                
            for n = 1:obj.num_softmax
                s = zeros(obj.num_softmax,N); s(n,:) = 1;
                E(n,:) = -obj.softmax_bias' * s - obj.visual_bias' * x - ... % ������������
                    sum(log(1 + exp(obj.weight_s2h * s + obj.weight_v2h * x + repmat(obj.hidden_bias,1,N))));
            end
            
            [~,y] = min(E);
            y = y - 1;
        end
    end
    
    methods (Access = private)
        function [d_w_s2h,d_w_v2h,d_h_bias,d_s_bias,d_v_bias,recon_error] = CD1(obj, minibatch, label)
            % ʹ��Contrastive Divergence 1 (CD1)������Լ����������RBM�����п���ѵ������RBM����Softmax��Ԫ��
            % ���룺
            %   minibatch��һ��ѵ�����ݣ�һ�д���һ��ѵ��������������ʾѵ�������ĸ���
            % ���:
            %   d_weight, Ȩֵ����ĵ�����ֵ.
            %   d_h_bias, ������Ԫƫ��ֵ�ĵ�����ֵ.
            %   d_v_bias, �ɼ���Ԫƫ��ֵ�ĵ�����ֵ.
            %   r_error,  �ؽ����ֵ
        
            N = size(minibatch,2); % ѵ�������ĸ���
            
            h_field_0 = obj.posterior(label, minibatch);
            h_state_0 = learn.sample(h_field_0);
            [s_field_1,v_field_1] = obj.likelihood(h_state_0);
            s_state_1 = learn.sample_softmax(s_field_1);
            v_state_1 = learn.sample(v_field_1);
            h_field_1 = obj.posterior(s_state_1,v_state_1); 
            
            recon_error =  sum(sum(([s_field_1;v_field_1] - [label;minibatch]).^2)) / N; %����������train_data�ϵ�ƽ��reconstruction error
            
            d_w_s2h  = (h_field_0 * label'     - h_field_1 * s_state_1') / N;
            d_w_v2h  = (h_field_0 * minibatch' - h_field_1 * v_state_1') / N;
            d_h_bias = (h_state_0 - h_field_1) * ones(N,1) / N;
            d_s_bias = (label     - s_field_1) * ones(N,1) / N;
            d_v_bias = (minibatch - v_field_1) * ones(N,1) / N;
        end
        
        function obj = initialize_weight(obj,train_data,train_label)
            %INTIALIZE ��ʼ��Ȩֵ����Ϊ0������С���������ʼ���Բ���Ԫ��ƫ��Ϊ������ʣ���ʼ��������Ԫ��ƫ��Ϊ0.
            obj.weight_s2h = 0.01 * randn(size(obj.weight_s2h));
            obj.weight_v2h = 0.01 * randn(size(obj.weight_v2h));
            
            obj.softmax_bias = mean(train_label,2);
            obj.visual_bias  = mean(train_data, 2);
            
            obj.softmax_bias = log(obj.softmax_bias ./ (1-obj.softmax_bias));
            obj.visual_bias  = log(obj.visual_bias  ./ (1-obj.visual_bias));
            
            obj.softmax_bias(obj.softmax_bias < -100) = -100;
            obj.softmax_bias(obj.softmax_bias > +100) = +100;
            
            obj.visual_bias(obj.visual_bias < -100) = -100;
            obj.visual_bias(obj.visual_bias > +100) = +100;
            
            obj.hidden_bias = zeros(size(obj.hidden_bias));
        end
    end
    
    methods(Static)
        function [srbm,e] = unit_test()
            clear all;
            close all;
            [data,~,test_data,test_label] = learn.import_mnist('./+learn/mnist.mat');
            [D,S,M] = size(data); N = S * M; K = 10;
            label = eye(10); label = repmat(label,1,10,M);
     
            srbm = learn.SoftmaxRBM(K,D,2000);
            srbm = srbm.initialize(data,label);
            
            parameters.learn_rate = [1e-6,1e-2];
            parameters.weight_cost = 1e-4;
            parameters.max_it = 1e6;
            srbm = srbm.pretrain(data,label,parameters);
            
            save('srbm.mat','srbm');
            % load('srbm.mat');
         
            y = srbm.classify(test_data);
            e = sum(y~=test_label') / length(test_label);
        end
    end
end


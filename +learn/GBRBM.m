classdef GBRBM
    %Gaussian Bernoulli Restricted Boltzmann Machine ��˹��Ŭ��Լ������������
    % �Բ���ԪΪ������Ԫ�Ӹ�˹������������ԪΪ��Ŭ����ֵ��Ԫ
    
    properties
        num_hidden;  % ������Ԫ�ĸ���
        num_visual;  % �ɼ���Ԫ�ĸ���
        weight;      % Ȩֵ����(num_hidden * num_visual)
        hidden_bias; % ������Ԫ��ƫ��
        visual_bias; % �ɼ���Ԫ��ƫ��
        visual_sgma; % �ɼ���Ԫ���Ը�˹������׼��
    end
    
    methods
        function obj = GBRBM(num_visual,num_hidden) % ���캯��
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
            %initialize ����ѵ�����ݣ��ɶ��minibatch��ɵ�ѵ�����ݼ��ϣ���ʼ��Ȩֵ��������Ԫƫ�ú�����Ԫƫ��
            %           minibatchs ��һ��[D,S,M]ά�����顣
           
            [D,S,M] = size(minibatchs);
            minibatchs = reshape(minibatchs,D,[]);
            obj = obj.initialize_weight(minibatchs);
        end

        function obj = pretrain(obj,minibatchs,learn_rate,learn_sgma,max_it) 
            %pretrain ��Ȩֵ����Ԥѵ��
            % ʹ��CD1�����㷨��Ȩֵ����Ԥѵ��
            
            [D,S,M] = size(minibatchs); % �õ�minibatch�ĸ���
            ob_window_size = M;     % �趨�۲촰�ڵĴ�СΪ
            ob_var_num = 1;                     % �趨�۲�����ĸ���
            ob = learn.Observer('�ؽ����',ob_var_num,ob_window_size,'xxx'); %��ʼ���۲��ߣ��۲��ؽ����
            
            % ��ʼ��velocity����
            v_weight = zeros(size(obj.weight));
            v_h_bias = zeros(size(obj.hidden_bias));
            v_v_bias = zeros(size(obj.visual_bias));
            v_v_sgma = zeros(size(obj.visual_sgma));
            
            % ��ʼ����������Ϊ0.5
            momentum = 0.5;
            
            r_error_list = zeros(1,ob_window_size);
            for idx = 1:M  % ��ʼ���ؽ�����б���ƶ�ƽ��ֵ
                minibatch = minibatchs(:,:,idx);
                [~, ~, ~, ~, r_error] = obj.CD1(minibatch);
                r_error_list(idx) = r_error;
            end
            r_error_ave_old = mean(r_error_list);
            ob = ob.initialize(r_error_ave_old);
            
            learn_rate_min = min(learn_rate);
            learn_rate     = max(learn_rate);  %��ʼ��ѧϰ�ٶ�
            
            for it = 0:max_it
                minibatch_idx = mod(it,M)+1;  % ȡһ��minibatch
                minibatch = minibatchs(:,:,minibatch_idx);
                
                [d_weight, d_h_bias, d_v_bias, d_v_sgma, r_error] = obj.CD1(minibatch);
                r_error_list(minibatch_idx) = r_error;
                r_error_ave_new = mean(r_error_list);
                
                if minibatch_idx == M % �����е�minibatch����Ѷ��һ���ʱ�򣨵���۲촰�����ұߵ�ʱ��
                    if r_error_ave_new > r_error_ave_old
                        learn_rate = learn_rate / 5;
                        if learn_rate < learn_rate_min
                            break;
                        end
                    end
                    r_error_ave_old = r_error_ave_new;
                end
                
                description = strcat('�ؽ����:',num2str(r_error_ave_new));
                description = strcat(description,strcat('��������:',num2str(it)));
                description = strcat(description,strcat('ѧϰ�ٶ�:',num2str(learn_rate)));
                % disp(description);
                ob = ob.showit(r_error_ave_new,description);
                
                momentum = min([momentum * 1.01,0.9]); % �����������Ϊ0.9����ʼֵΪ0.5����Լ����60��֮�������ʴﵽ0.9��
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
            N = size(x,2); % �����ĸ���
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
            v_sgma = repmat(obj.visual_sgma,1,N);
            
            h_field_0 = learn.sigmoid(obj.weight * (minibatch ./ v_sgma) + h_bias);
            h_state_0 = learn.sample(h_field_0);
            v_field_1 = v_sgma .* (obj.weight'* h_state_0) + v_bias;
            v_state_1 = v_field_1 + v_sgma .* randn(size(v_field_1));
            h_field_1 = learn.sigmoid(obj.weight * (v_state_1 ./ v_sgma) + h_bias);
            h_state_1 = learn.sample(h_field_1);
            
            r_error =  sum(sum((v_field_1 - minibatch).^2)) / N; %����������minibatch�ϵ�ƽ���ؽ����
            
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
            %INTIALIZE ��ʼ��Ȩֵ����Ϊ0������С���������ʼ���Բ���Ԫ��ƫ��Ϊ������ʣ���ʼ��������Ԫ��ƫ��Ϊ0.
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


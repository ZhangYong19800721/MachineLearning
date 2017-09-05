classdef SAE < learn.StackedRBM
    %STACKED AUTO ENCODER ջʽ�Զ�������
    %   
    
    methods
        function obj = SAE(configure) % ���캯��
            obj@learn.StackedRBM(configure); % ���ø���Ĺ��캯��
        end
    end
    
    methods
        function obj = train(obj,minibatchs,parameters)
            switch parameters.case
                case 1 % ȫ���������
                    obj = obj.train_case1(minibatchs,parameters);
                case 2 % �޳��������
                    obj = obj.train_case2(minibatchs,parameters);
                otherwise
                    error('parameters.case��ֵ����')
            end
        end
        
        function code = encode(obj,data,option)
            %ENCODE �������ݣ���������� 
            %
            data = obj.posterior(data);
            if strcmp(option,'sample')
                code = learn.sample(data{length(data)});
            elseif strcmp(option,'nosample')
                code = data{length(data)};
            elseif strcmp(option,'fix')
                code = data{length(data)} > 0.5;
            else
                error('option��ֵ����');
            end
        end
        
        function data = decode(obj,code)
            %DECODE �������룬����������
            %   
            data = obj.likelihood(code);
            data = data{1};
        end
        
        function rebuild_data = rebuild(obj,data,option)
            %REBUILD ����ԭ���ݣ�ͨ���������������ؽ�����
            %      
            rebuild_data = obj.decode(obj.encode(data,option));
        end
    end
    
    methods(Access = private)
        function obj = train_case1(obj,minibatchs,parameters) 
            % train ѵ������
            % ʹ��UPDOWN�㷨����ѵ����ȫ����
            obj = obj.weightsync(); % Ȩֵͬ����������
            
            [D,S,M] = size(minibatchs); L = obj.layer_num(); % D����ά�ȣ�S���Ĵ�С��M���ĸ�����L��ĸ���
            ob = learn.Observer('�ؽ����',1,M); %��ʼ���۲��������۲��ؽ����
                        
            recon_error_list = zeros(1,M);
            for m = 1:M
                minibatch = minibatchs(:,:,m);
                recon_minibatch = obj.rebuild(minibatch,'sample');
                recon_error_list(m) = sum(sum((recon_minibatch - minibatch).^2)) / S;
            end
            
            recon_error_ave_old = mean(recon_error_list); % �����ؽ����ľ�ֵ
            ob = ob.initialize(recon_error_ave_old);      % �þ�ֵ��ʼ���۲���
            
            learn_rate_min = min(parameters.learn_rate);
            learn_rate     = max(parameters.learn_rate);  % ��ѧϰ�ٶȳ�ʼ��Ϊ���ѧϰ�ٶ�
            
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
                
                % ���²���Ȩֵ
                for l = 1:L
                    obj.rbms{l}.weight_h2v = obj.rbms{l}.weight_h2v + learn_rate * ...
                        pos_state{l+1}.state * (pos_state{l}.state - pre_pos_proba{l})' / S;
                    obj.rbms{l}.visual_bias = obj.rbms{l}.visual_bias + learn_rate * ...
                        sum(pos_state{l}.state - pre_pos_proba{l},2) / S;
                end
                
                % ����ʶ��Ȩֵ
                for l = 1:L
                    obj.rbms{l}.weight_v2h = obj.rbms{l}.weight_v2h + learn_rate * ...
                        (neg_state{l+1}.state - pre_neg_proba{l+1}) * neg_state{l}.state' / S;
                    obj.rbms{l}.hidden_bias = obj.rbms{l}.hidden_bias + learn_rate * ...
                        sum(neg_state{l+1}.state - pre_neg_proba{l+1},2) / S;
                end
                
                % �����ؽ����
                recon_minibatch = obj.rebuild(minibatch,'sample');
                recon_error = sum(sum((recon_minibatch - minibatch).^2)) / S;
                recon_error_list(m) = recon_error;
                recon_error_ave_new = mean(recon_error_list);
                
                if m == M
                    if recon_error_ave_new > recon_error_ave_old % ������M�ε�����ƽ���ؽ����½�ʱ
                        learn_rate = learn_rate / 2;         % ������ѧϰ�ٶ�
                        if learn_rate < learn_rate_min       % ��ѧϰ�ٶ�С����Сѧϰ�ٶ�ʱ���˳�
                            break;
                        end
                    end
                    recon_error_ave_old = recon_error_ave_new;
                end
                
                % ��ͼ
                description = strcat('�ؽ���',num2str(recon_error_ave_new));
                description = strcat(description,strcat('��������:',num2str(it)));
                description = strcat(description,strcat('ѧϰ�ٶ�:',num2str(learn_rate)));
                % disp(description);
                ob = ob.showit(recon_error_ave_new,description);
            end
        end
        
        function obj = train_case2(obj,minibatchs,parameters) 
            % train_case2 ѵ������
            % ʹ��UPDOWN�㷨����ѵ�����޳���
            
            %% Ȩֵͬ����������
            obj = obj.weightsync();
            
            %% D����ά�ȣ�S���Ĵ�С��M���ĸ�����L��ĸ���
            [D,S,M] = size(minibatchs); L = obj.layer_num(); 
            ob = learn.Observer('�ؽ����',1,M); %��ʼ���۲��������۲��ؽ����
                        
            recon_error_list = zeros(1,M);
            for m = 1:M
                minibatch = minibatchs(:,:,m);
                recon_minibatch = obj.rebuild(minibatch,'nosample');
                recon_error_list(m) = sum(sum((recon_minibatch - minibatch).^2)) / S;
            end
            
            recon_error_ave_old = mean(recon_error_list); % �����ؽ����ľ�ֵ
            ob = ob.initialize(recon_error_ave_old);      % �þ�ֵ��ʼ���۲���
            
            learn_rate_min = min(parameters.learn_rate);
            learn_rate     = max(parameters.learn_rate);  % ��ѧϰ�ٶȳ�ʼ��Ϊ���ѧϰ�ٶ�
            
            %% ��ʼ����������
            % ��ʼ����������Ϊ0.5
            momentum = 0.5;
            
            % ��ʼ������ֵ
            inc = cell(1,L);
            for l = 1:L
                inc{l}.weight_v2h = zeros(size(obj.rbms{l}.weight_v2h));
                inc{l}.weight_h2v = zeros(size(obj.rbms{l}.weight_h2v));
                inc{l}.visual_bias = zeros(size(obj.rbms{l}.visual_bias));
                inc{l}.hidden_bias = zeros(size(obj.rbms{l}.hidden_bias));
            end
            
            %% ��ʼ����
            for it = 0:parameters.max_it
                m = mod(it,M) + 1;
                minibatch = minibatchs(:,:,m);
                
                pos_proba = obj.posterior(minibatch);
                neg_proba = obj.likelihood(pos_proba{L+1});
                
                for l = L:-1:1
                    pre_pos_proba{l  } = obj.rbms{l}.likelihood(pos_proba{l+1});
                    pre_neg_proba{l+1} = obj.rbms{l}.posterior (neg_proba{l  });
                end
                
                momentum = min([momentum * 1.01,0.9]); % �����������Ϊ0.9����ʼֵΪ0.5����Լ����60��֮�������ʴﵽ0.9��
                
                for l = 1:L
                    inc{l}.weight_h2v = momentum * inc{l}.weight_h2v + learn_rate * pos_proba{l+1} * (pos_proba{l} - pre_pos_proba{l})' / S;
                    inc{l}.visual_bias = momentum * inc{l}.visual_bias + learn_rate * sum(pos_proba{l} - pre_pos_proba{l},2) / S;
                    inc{l}.weight_v2h = momentum * inc{l}.weight_v2h + learn_rate * (neg_proba{l+1} - pre_neg_proba{l+1}) * neg_proba{l}' / S;
                    inc{l}.hidden_bias = momentum * inc{l}.hidden_bias + learn_rate * sum(neg_proba{l+1} - pre_neg_proba{l+1},2) / S;
                end
                
                %% ���²���Ȩֵ
                for l = 1:L
                    obj.rbms{l}.weight_h2v = obj.rbms{l}.weight_h2v + inc{l}.weight_h2v;
                    obj.rbms{l}.visual_bias = obj.rbms{l}.visual_bias + inc{l}.visual_bias;
                end
                
                %% ����ʶ��Ȩֵ
                for l = 1:L
                    obj.rbms{l}.weight_v2h = obj.rbms{l}.weight_v2h + inc{l}.weight_v2h;
                    obj.rbms{l}.hidden_bias = obj.rbms{l}.hidden_bias + inc{l}.hidden_bias;
                end
                
                %% �����ؽ����
                recon_minibatch = obj.rebuild(minibatch,'nosample');
                recon_error = sum(sum((recon_minibatch - minibatch).^2)) / S;
                recon_error_list(m) = recon_error;
                recon_error_ave_new = mean(recon_error_list);
                
                if m == M
                    if recon_error_ave_new > recon_error_ave_old % ������M�ε�����ƽ���ؽ����½�ʱ
                        learn_rate = learn_rate / 2;         % ������ѧϰ�ٶ�
                        if learn_rate < learn_rate_min       % ��ѧϰ�ٶ�С����Сѧϰ�ٶ�ʱ���˳�
                            break;
                        end
                    end
                    recon_error_ave_old = recon_error_ave_new;
                end
                
                %% ��ͼ
                description = strcat('�ؽ���',num2str(recon_error_ave_new));
                description = strcat(description,strcat('��������:',num2str(it)));
                description = strcat(description,strcat('ѧϰ�ٶ�:',num2str(learn_rate)));
                %disp(description);
                ob = ob.showit(recon_error_ave_new,description);
            end
        end
    end
    
    methods(Static)
        function [sae,e] = unit_test()
            clear all;
            close all;
            rng(1);
            
            [data,~,~,~] = learn.import_mnist('./+learn/mnist.mat');
            [D,S,M] = size(data); N = S * M;
            
            configure = [D,500,256];
            sae = learn.SAE(configure);
            
            parameters.learn_rate = [1e-6,1e-2];
            parameters.weight_cost = 1e-4;
            parameters.max_it = 1e0;
            sae = sae.pretrain(data,parameters);
            % save('sae_mnist_pretrain.mat','sae');
            load('sae_mnist_pretrain.mat');
            
            data = reshape(data,D,[]);
            recon_data1 = sae.rebuild(data,'sample');
            error1 = sum(sum((recon_data1 - data).^2)) / N
            
            data = reshape(data,D,S,M);
            parameters.max_it = 1e6;
            parameters.case = 1;
            sae = sae.train(data,parameters);
            
            save('sae_mnist_finetune2.mat','sae');
            
            data = reshape(data,D,[]);
            recon_data2 = sae.rebuild(data,'sample');
            error2 = sum(sum((recon_data2 - data).^2)) / N
            e = error2;
        end
    end
end

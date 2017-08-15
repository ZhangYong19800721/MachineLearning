classdef SAE
    %STACKED AUTO ENCODER ջʽ�Զ�������
    %   
    
    properties
        encoder;
        decoder;
    end
    
    methods
        function obj = SAE(configure) 
            % ���캯��
            obj.encoder = learn.StackedRBM(configure);
            obj.decoder = obj.encoder;
        end
    end
    
    methods
        function obj = pretrain(obj,minibatchs,parameters) 
            %pretrain ʹ��CD1�����㷨�����ѵ��
            obj.encoder = obj.encoder.pretrain(minibatchs,parameters); % ��encoder����Ԥѵ��
            obj.decoder = obj.encoder; % ��decoder��ʼ��Ϊ��encoder��ͬ��Ȩֵ������
        end
        
        function obj = train(obj,minibatchs,parameters) 
            % train ѵ������
            % ʹ��wake-sleep�㷨����ѵ��
            
            [D,S,M] = size(minibatchs); L = obj.decoder.layer_num(); % D����ά�ȣ�S���Ĵ�С��M���ĸ�����L��ĸ���
            ob = learn.Observer('�ؽ����',1,M); %��ʼ���۲��������۲��ؽ����
                        
            recon_error_list = zeros(1,M);
            for m = 1:M
                minibatch = minibatchs(:,:,m);
                recon_minibatch = obj.rebuild(minibatch);
                recon_error_list(m) = sum(sum((recon_minibatch - minibatch).^2)) / S;
            end
            
            recon_error_ave_old = mean(recon_error_list); % �����ؽ����ľ�ֵ
            ob = ob.initialize(recon_error_ave_old);      % �þ�ֵ��ʼ���۲���
            
            learn_rate_min = min(parameters.learn_rate);
            learn_rate     = max(parameters.learn_rate);  % ��ѧϰ�ٶȳ�ʼ��Ϊ���ѧϰ�ٶ�
            
            for it = 0:parameters.max_it
                m = mod(it,M) + 1;
                minibatch = minibatchs(:,:,m);
                
                pos_state = obj.encoder.posterior_sample(minibatch);
                neg_state = obj.decoder.likelihood_sample(pos_state{L+1}.state);
                neg_state{1}.state = neg_state{1}.proba;
                
                for l = 1:L
                    pre_pos_proba{l} = obj.decoder.rbms{l}.likelihood(pos_state{l+1}.state);
                    pre_neg_proba{l+1} = obj.encoder.rbms{l}.posterior(neg_state{l}.state);
                end
                
                % ���²���Ȩֵ
                for l = 1:L
                    obj.decoder.rbms{l}.weight = obj.decoder.rbms{l}.weight + learn_rate * ...
                        pos_state{l+1}.state * (pos_state{l}.state - pre_pos_proba{l})' / S;
                    obj.decoder.rbms{l}.visual_bias = obj.decoder.rbms{l}.visual_bias + learn_rate * ...
                        sum(pos_state{l}.state - pre_pos_proba{l},2) / S;
                end
                
                % ����ʶ��Ȩֵ
                for l = 1:L
                    obj.encoder.rbms{l}.weight = obj.encoder.rbms{l}.weight + learn_rate * ...
                        (neg_state{l+1}.state - pre_neg_proba{l+1}) * neg_state{l}.state' / S;
                    obj.encoder.rbms{l}.hidden_bias = obj.encoder.rbms{l}.hidden_bias + learn_rate * ...
                        sum(neg_state{l+1}.state - pre_neg_proba{l+1},2) / S;
                end
                
                % �����ؽ����
                recon_minibatch = obj.rebuild(minibatch);
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
        
        function code = encode(obj,data)
            %ENCODE �������ݣ���������� 
            %
            data = obj.encoder.posterior(data);
            code = learn.sample(data);
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


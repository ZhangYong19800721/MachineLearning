classdef DBN
%Deep Belief Net ����Ŷ�����   
%
    properties
        stacked_rbm;
        softmax_rbm;
    end
    
    methods
        function obj = DBN(configure) 
            % DeepBeliefNet ���캯��
            obj.stacked_rbm = learn.StackedRBM(configure.stacked_rbm);
            obj.softmax_rbm = learn.SoftmaxRBM(configure.softmax_rbm(1),configure.softmax_rbm(2),configure.softmax_rbm(3));
        end
    end
    
    methods        
        function obj = pretrain(obj,minibatchs,labels,parameters)
            %pretrain Ԥѵ��
            % ʹ��CD1�����㷨�����ѵ��Լ��������������RBM��           
            [D,S,M] = size(minibatchs); 
            obj.stacked_rbm = obj.stacked_rbm.pretrain(minibatchs,parameters); % Ԥѵ��ջʽRBM     
            minibatchs = reshape(minibatchs,D,[]);
            minibatchs = obj.stacked_rbm.posterior(minibatchs);
            minibatchs = reshape(minibatchs,[],S,M);
            obj.softmax_rbm = obj.softmax_rbm.initialize(minibatchs,labels);
            obj.softmax_rbm = obj.softmax_rbm.pretrain(minibatchs,labels,parameters);
        end
        
        function obj = train(obj,minibatchs,labels,parameters)
            %train UP-DOWNѵ���㷨
            %           
            [D,S,M] = size(minibatchs); [K,~,~] = size(labels); L = obj.stacked_rbm.layer_num();
            ob = learn.Observer('�ؽ����',1,M); %��ʼ���۲��ߣ��۲��ؽ����
            
            obj.stacked_rbm = obj.stacked_rbm.weightsync(); % Ȩֵͬ��/����
            % obj.softmax_rbm = obj.softmax_rbm.weightsync(); % Ȩֵͬ��/����
            
            learn_rate_min = min(parameters.learn_rate);
            learn_rate     = max(parameters.learn_rate);
            
            recon_error_list = zeros(1,M);
            for m = 1:M  % ��ʼ���ؽ�����б���ƶ�ƽ��ֵ
                minibatch = minibatchs(:,:,m); label = labels(:,:,m);
                [recon_label,recon_data] = obj.rebuild(label,minibatch);
                recon_error = sum(sum(([recon_label;recon_data] - [label;minibatch]).^2)) / S;
                recon_error_list(m) = recon_error;
            end
            recon_error_ave_old = mean(recon_error_list);
            ob = ob.initialize(recon_error_ave_old);
            
            for it = 0:parameters.max_it
                m = mod(it,M)+1;
                label = labels(:,:,m);
                minibatch = minibatchs(:,:,m);
                
                [recon_label,recon_data] = obj.rebuild(label,minibatch);
                recon_error = sum(sum(([recon_label;recon_data] - [label;minibatch]).^2)) / S;
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
                ob = ob.showit(recon_error_ave_new,description);
                
                % positive phase (wake �׶�)
                pos_state = obj.stacked_rbm.posterior_sample(minibatch);
                pos_top_hid_proba = obj.softmax_rbm.posterior(label,pos_state{L+1}.state);
                pos_top_hid_state = learn.sample(pos_top_hid_proba);
                
                % CD1
                neg_top_hid_state = pos_top_hid_state;
                [neg_top_lab_proba,neg_top_vis_proba] = obj.softmax_rbm.likelihood(neg_top_hid_state);
                neg_top_lab_state = learn.sample_softmax(neg_top_lab_proba);
                neg_top_vis_state = neg_top_vis_proba;
                neg_top_hid_proba = obj.softmax_rbm.posterior(neg_top_lab_state, neg_top_vis_state);
                neg_top_hid_state = learn.sample(neg_top_hid_proba);
                
                % negative phase (sleep �׶�)
                neg_state = obj.stacked_rbm.likelihood_sample(neg_top_vis_state);
                neg_state{1}.state = neg_state{1}.proba;
                
                % prediction
                for l = L:-1:1
                    pre_neg_proba{l+1} = obj.stacked_rbm.rbms{l}.posterior (neg_state{l  }.state);
                    pre_pos_proba{l  } = obj.stacked_rbm.rbms{l}.likelihood(pos_state{l+1}.state);
                end
                
                % ���²���Ȩֵ
                for l = 1:L
                    obj.stacked_rbm.rbms{l}.weight_h2v  = obj.stacked_rbm.rbms{l}.weight_h2v  + learn_rate * pos_state{l+1}.state * (pos_state{l}.state - pre_pos_proba{l})' / S;
                    obj.stacked_rbm.rbms{l}.visual_bias = obj.stacked_rbm.rbms{l}.visual_bias + learn_rate * sum(pos_state{l}.state - pre_pos_proba{l},2) / S; 
                end
                
                % ���¶���Ȩֵ
                obj.softmax_rbm.weight_s2h   = obj.softmax_rbm.weight_s2h   + learn_rate * (       pos_top_hid_state  * label'                -        neg_top_hid_state  * neg_top_lab_state') / S;
                obj.softmax_rbm.weight_v2h   = obj.softmax_rbm.weight_v2h   + learn_rate * (double(pos_top_hid_state) * pos_state{L+1}.state' - double(neg_top_hid_state) * neg_top_vis_state') / S;
                obj.softmax_rbm.softmax_bias = obj.softmax_rbm.softmax_bias + learn_rate * sum(label                - neg_top_lab_state,2) / S;
                obj.softmax_rbm.visual_bias  = obj.softmax_rbm.visual_bias  + learn_rate * sum(pos_state{L+1}.state - neg_top_vis_state,2) / S;
                obj.softmax_rbm.hidden_bias  = obj.softmax_rbm.hidden_bias  + learn_rate * sum(pos_top_hid_state    - neg_top_hid_state,2) / S;
                
                % ����ʶ��Ȩֵ
                for l = 1:L
                    obj.stacked_rbm.rbms{l}.weight_v2h  = obj.stacked_rbm.rbms{l}.weight_v2h  + learn_rate * (neg_state{l+1}.state - pre_neg_proba{l+1}) * neg_state{l}.state' / S;
                    obj.stacked_rbm.rbms{l}.hidden_bias = obj.stacked_rbm.rbms{l}.hidden_bias + learn_rate * sum(neg_state{l+1}.state - pre_neg_proba{l+1},2) / S;
                end
            end
        end
        
        function [c,y] = rebuild(obj,s,x)
            x = obj.stacked_rbm.posterior(x);
            [c,z] = obj.softmax_rbm.rebuild(s,x);
            y = obj.stacked_rbm.likelihood(z);
        end
        
        function y = classify(obj,x)
            x = obj.stacked_rbm.posterior(x);
            y = obj.softmax_rbm.classify(x);
        end
    end
    
    methods(Static)
        function [dbn,e] = unit_test()
            clear all;
            close all;
            rng(1);
            
            [train_images,~,test_images,test_labels] = learn.import_mnist('./+learn/mnist.mat');
            K = 10; [D,S,M] = size(train_images);
            train_labels = eye(10); train_labels = repmat(train_labels,1,10,M);
            
            configure.stacked_rbm = [D,500,500];
            configure.softmax_rbm = [K,500,2000];
            dbn = learn.DBN(configure);
            
            parameters.learn_rate = [1e-6 1e-2];
            parameters.weight_cost = 1e-4;
            parameters.max_it = 1e6;
            dbn = dbn.pretrain(train_images,train_labels,parameters);
            
            save('dbn_pretrain.mat','dbn');
            %load('dbn_pretrain.mat');
            
            y = dbn.classify(test_images);
            error1 = sum(y~=test_labels') / length(y);
            
            parameters.max_it = 1e6;
            dbn = dbn.train(train_images,train_labels,parameters);
            save('dbn_train.mat','dbn');
            y = dbn.classify(test_images);
            error2 = sum(y~=test_labels') / length(y);
        end
    end
end


classdef DBN
%Deep Belief Net 深度信度网络   
%
    properties
        stacked_rbm;
        softmax_rbm;
    end
    
    methods
        function obj = DBN(configure) 
            % DeepBeliefNet 构造函数
            obj.stacked_rbm = learn.StackedRBM(configure.stacked_rbm);
            obj.softmax_rbm = learn.SoftmaxRBM(configure.softmax_rbm(1),configure.softmax_rbm(2),configure.softmax_rbm(3));
        end
    end
    
    methods        
        function h_field = posterior(obj,v_state)
            % posterior 给定显层神经元的取值，计算顶层隐藏神经元的域值
            h_field = v_state;
            for layer_idx = 1:length(obj.rbm_layers)
                h_field = obj.rbm_layers{layer_idx}.posterior(h_field);
            end
        end
        
        function v_field = likelihood(obj,h_state)
            % likelihood 给定顶层隐神经元的取值，计算底层显神经元的域值
            v_field = h_state;
            for layer_idx = length(obj.rbm_layers):-1:1
                v_field = obj.rbm_layers{layer_idx}.likelihood(v_field);
            end
        end
        
        function obj = pretrain(obj,minibatchs,labels,parameters)
            %pretrain 预训练
            % 使用CD1快速算法，逐层训练约束玻尔兹曼机（RBM）           
            [D,S,M] = size(minibatchs); 
            obj.stacked_rbm = obj.stacked_rbm.pretrain(minibatchs,parameters); % 预训练栈式RBM     
            minibatchs = reshape(minibatchs,D,[]);
            minibatchs = obj.stacked_rbm.posterior(minibatchs);
            minibatchs = reshape(minibatchs,[],S,M);
            obj.softmax_rbm = obj.softmax_rbm.initialize(minibatchs,labels);
            obj.softmax_rbm = obj.softmax_rbm.pretrain(minibatchs,labels,parameters);
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
            
            [train_images,~,test_images,test_labels] = learn.import_mnist('./+learn/mnist.mat');
            K = 10; [D,S,M] = size(train_images); 
            train_labels = eye(10); train_labels = repmat(train_labels,1,10,M);
            
            configure.stacked_rbm = [D,500,500];
            configure.softmax_rbm = [K,500,2000];
            dbn = learn.DBN(configure);
            
            parameters.learn_rate = [1e-8 1e-2];
            parameters.weight_cost = 1e-4;
            parameters.max_it = 1e6;
            dbn = dbn.pretrain(train_images,train_labels,parameters);
            
            save('dbn.mat','dbn');
            
            y = dbn.classify(test_images);
            e = sum(y~=test_labels') / length(y);
        end
    end
end


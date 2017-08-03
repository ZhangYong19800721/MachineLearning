classdef DeepBeliefNet
%DeepBeliefNet 深度信度网络   
%
    properties
        stacked_rbm;
        softmax_rbm;
    end
    
    methods
        function obj = DeepBeliefNet(configure) 
            % DeepBeliefNet 构造函数
            obj.stacked_rbm = learn.StackedRestrictedBoltzmannMachine(configure.stacked_rbm);
            obj.softmax_rbm = learn.SoftmaxRestrictedBoltzmannMachine(configure.softmax_rbm(1),configure.softmax_rbm(2),configure.softmax_rbm(3));
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
        
        function obj = pretrain(obj,minibatchs,init_visual_bias,init_hidden_bias,learn_rate_min,learn_rate_max,max_it)
            %pretrain 预训练
            % 使用CD1快速算法，逐层训练约束玻尔兹曼机（RBM）
            num_softmax = obj.softmax_rbm.num_softmax;
            num_visual  = size(minibatchs{1},1);
            
            for minibatch_idx = 1:length(minibatchs)
                minibatchs_label{minibatch_idx} = minibatchs{minibatch_idx}(1:num_softmax,:);
                minibatchs_point{minibatch_idx} = minibatchs{minibatch_idx}((1+num_softmax):num_visual,:);
            end
            
            obj.stacked_rbm = obj.stacked_rbm.pretrain(minibatchs_point,init_visual_bias,init_hidden_bias,learn_rate_min,learn_rate_max,max_it); % 预训练栈式RBM
            
            for minibatch_idx = 1:length(minibatchs)
                minibatchs_point{minibatch_idx} = obj.stacked_rbm.posterior(minibatchs_point{minibatch_idx});
                minibatchs{minibatch_idx} = [minibatchs_label{minibatch_idx};minibatchs_point{minibatch_idx}];
            end
            
            obj.softmax_rbm = obj.softmax_rbm.pretrain(minibatchs,learn_rate_min,learn_rate_max,max_it);
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
            [train_images,train_labels,test_images,test_labels] = learn.import_mnist('./+learn/mnist.mat');
            [D,minibatch_size,minibatch_num] = size(train_images); K = 10;
            for minibatch_idx = 1:minibatch_num
                L = zeros(K,minibatch_size);
                I = sub2ind(size(L),1+train_labels(:,minibatch_idx),[1:minibatch_size]');
                L(I) = 1;
                mnist{minibatch_idx} = [L;train_images(:,:,minibatch_idx)];
            end
            
            configure.stacked_rbm = [D,500,500];
            configure.softmax_rbm = [K,500,2000];
            
            dbn = learn.DeepBeliefNet(configure);
            dbn = dbn.pretrain(mnist,-6,-4,1e-6,0.1,1e6);
            
            save('dbn.mat','dbn');
            
            y = dbn.classify(test_images);
            e = sum(y~=test_labels') / length(y);
        end
    end
end


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
            obj.stacked_rbm = ML.StackedRestrictedBoltzmannMachine(configure.stacked_rbm);
            obj.softmax_rbm = ML.SoftmaxRestrictedBoltzmannMachine(configure.softmax_rbm(1),configure.softmax_rbm(2),configure.softmax_rbm(3));
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
        
        function obj = pretrain(obj,minibatchs,learn_rate_min,learn_rate_max,max_it)
            %pretrain 预训练
            % 使用CD1快速算法，逐层训练约束玻尔兹曼机（RBM）
            num_softmax = obj.softmax_rbm.num_softmax;
            num_visual  = size(minibatchs{1},1);
            
            for minibatch_idx = 1:length(minibatchs)
                minibatchs_label{minibatch_idx} = minibatchs{minibatch_idx}(1:num_softmax,:);
                minibatchs_point{minibatch_idx} = minibatchs{minibatch_idx}((1+num_softmax):num_visual,:);
            end
            
            obj.stacked_rbm = obj.stacked_rbm.pretrain(minibatchs_point,learn_rate_min,learn_rate_max,max_it); % 预训练栈式RBM
            
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
end


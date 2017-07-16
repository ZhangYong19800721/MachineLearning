classdef StackedRestrictedBoltzmannMachine
%StackedRestrictedBoltzmannMachine 堆叠玻尔兹曼机   
%
    properties
        rbm_layers;
    end
    
    methods
        function obj = StackedRestrictedBoltzmannMachine(configure) 
            % DeepBeliefNet 构造函数
            layer_num = length(configure) - 1;
            for layer_idx = 1:layer_num
                obj.rbm_layers{layer_idx} = ML.RestrictedBoltzmannMachine(configure(layer_idx),configure(layer_idx+1));
            end
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
            layer_num = length(obj.rbm_layers);
            
            for layer_idx = 1:layer_num
                obj.rbm_layers{layer_idx} = obj.rbm_layers{layer_idx}.initialize(minibatchs); % 初始化第layer_idx层的RBM
                obj.rbm_layers{layer_idx} = obj.rbm_layers{layer_idx}.pretrain(minibatchs,learn_rate_min,learn_rate_max,max_it); % 训练第layer_idx层的RBM
                for minibatch_idx = 1:length(minibatchs) %将训练数据映射到上一层
                    minibatchs{minibatch_idx} = obj.rbm_layers{layer_idx}.posterior(minibatchs{minibatch_idx});
                end
            end
        end
    end
end


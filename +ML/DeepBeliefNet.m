classdef DeepBeliefNet
%DeepBeliefNet 深度信度网络   
%
    properties
        rbm_layers;
    end
    
    methods
        function obj = DeepBeliefNet(configure)
            layer_num = length(configure) - 1;
            for layer_idx = 1:layer_num
                obj.rbm_layers{layer_idx} = ML.RestrictedBoltzmanMachine(configure(layer_idx),configure(layer_idx+1));
            end
        end
    end
    
    methods
        obj = train_sgd(obj,directory,filename,minibatch_num,learn_rate_min,learn_rate_max,max_iteration) % 使用CD1快速算法，逐层训练DBN
        type = discriminate(obj,data) % 给定显神经元的取值，识别其类别 
        data = recall(obj,type) % 给定softmax神经元的取值，回忆数据
        
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


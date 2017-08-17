classdef StackedRBM
%Stacked Restricted Boltzmann Machine 堆叠玻尔兹曼机   
%
    properties
        rbms;
    end
    
    methods
        function obj = StackedRBM(configure) 
            % DeepBeliefNet 构造函数
            L = length(configure) - 1;
            for l = 1:L
                obj.rbms{l} = learn.RBM(configure(l),configure(l+1));
            end
        end
    end
    
    methods        
        function proba = posterior(obj,v_state)
            % posterior 给定显层神经元的取值，计算顶层隐藏神经元的域值
            L = obj.layer_num(); proba = cell(1,L);
            
            proba{1} = v_state;
            for l = 1:L
                proba{l+1} = obj.rbms{l}.posterior(proba{l});
            end
        end
        
        function state = posterior_sample(obj,v_state)
            % posterior_sample 给定显层神经元的状态，计算顶层隐藏神经元的状态
            L = obj.layer_num();
            state = cell(1,L+1);
            
            state{1}.proba = v_state;
            state{1}.state = v_state;
            
            for l = 2:(L+1)
                state{l}.proba = obj.rbms{l-1}.posterior(state{l-1}.state);
                state{l}.state = learn.sample(state{l}.proba);
            end
        end
        
        function proba = likelihood(obj,h_state)
            % likelihood 给定顶层隐神经元的取值，计算底层显神经元的域值
            L = obj.layer_num();
            
            proba{L+1} = h_state;
            for l = L:-1:1
                proba{l} = obj.rbms{l}.likelihood(proba{l+1});
            end
        end
        
        function state = likelihood_sample(obj,h_state)
            % likelihood_sample 给定顶层神经元的状态，计算底层隐藏神经元的状态
            L = obj.layer_num();
            
            state{L+1}.proba = h_state;
            state{L+1}.state = h_state;

            for l = L:-1:1
                state{l}.proba = obj.rbms{l}.likelihood(state{l+1}.state);
                state{l}.state = learn.sample(state{l}.proba);
            end
        end
        
        function obj = pretrain(obj,minibatchs,parameters)
            %pretrain 预训练
            % 使用CD1快速算法，逐层训练约束玻尔兹曼机（RBM）
            L = obj.layer_num();
            [D,S,M] = size(minibatchs);
            
            for l = 1:L
                obj.rbms{l} = obj.rbms{l}.initialize(minibatchs); % 初始化第l层的RBM
                obj.rbms{l} = obj.rbms{l}.pretrain(minibatchs,parameters); % 训练第l层的RBM
                minibatchs = reshape(minibatchs,obj.rbms{l}.num_visual,[]);
                minibatchs = obj.rbms{l}.posterior(minibatchs);
                minibatchs = reshape(minibatchs,obj.rbms{l}.num_hidden,S,M);
            end
        end
        
        function l = layer_num(obj)
            l = length(obj.rbms);
        end
        
        function obj = weightsync(obj)
            L = obj.layer_num(); % 层数
            for l = 1:L
                obj.rbms{l} = obj.rbms{l}.weightsync();
            end
        end
    end
end


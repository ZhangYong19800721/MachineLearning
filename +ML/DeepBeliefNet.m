classdef DeepBeliefNet
%DeepBeliefNet ����Ŷ�����   
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
        obj = train_sgd(obj,directory,filename,minibatch_num,learn_rate_min,learn_rate_max,max_iteration) % ʹ��CD1�����㷨�����ѵ��DBN
        type = discriminate(obj,data) % ��������Ԫ��ȡֵ��ʶ������� 
        data = recall(obj,type) % ����softmax��Ԫ��ȡֵ����������
        
        function obj = pretrain(obj,minibatchs,learn_rate_min,learn_rate_max,max_it)
            %pretrain Ԥѵ��
            % ʹ��CD1�����㷨�����ѵ��Լ��������������RBM��
            layer_num = length(obj.rbm_layers);
            
            for layer_idx = 1:layer_num
                obj.rbm_layers{layer_idx} = obj.rbm_layers{layer_idx}.initialize(minibatchs); % ��ʼ����layer_idx���RBM
                obj.rbm_layers{layer_idx} = obj.rbm_layers{layer_idx}.pretrain(minibatchs,learn_rate_min,learn_rate_max,max_it); % ѵ����layer_idx���RBM
                for minibatch_idx = 1:length(minibatchs) %��ѵ������ӳ�䵽��һ��
                    minibatchs{minibatch_idx} = obj.rbm_layers{layer_idx}.posterior(minibatchs{minibatch_idx});
                end
            end
        end
    end
end


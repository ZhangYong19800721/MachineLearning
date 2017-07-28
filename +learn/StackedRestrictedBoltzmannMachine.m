classdef StackedRestrictedBoltzmannMachine
%StackedRestrictedBoltzmannMachine �ѵ�����������   
%
    properties
        rbm_layers;
    end
    
    methods
        function obj = StackedRestrictedBoltzmannMachine(configure) 
            % DeepBeliefNet ���캯��
            layer_num = length(configure) - 1;
            for layer_idx = 1:layer_num
                obj.rbm_layers{layer_idx} = ML.RestrictedBoltzmannMachine(configure(layer_idx),configure(layer_idx+1));
            end
        end
    end
    
    methods        
        function h_field = posterior(obj,v_state)
            % posterior �����Բ���Ԫ��ȡֵ�����㶥��������Ԫ����ֵ
            h_field = v_state;
            for layer_idx = 1:length(obj.rbm_layers)
                h_field = obj.rbm_layers{layer_idx}.posterior(h_field);
            end
        end
        
        function [t_state,state] = posterior_sample(obj,v_state)
            % posterior_sample �����Բ���Ԫ��״̬�����㶥��������Ԫ��״̬
            state{1}.v_field = v_state;
            state{1}.v_state = v_state;
            
            for layer_idx = 1:length(obj.rbm_layers)
                state{layer_idx}.h_field = obj.rbm_layers{layer_idx}.posterior(state{layer_idx}.v_state);
                state{layer_idx}.h_state = ML.sample(state{layer_idx}.h_field);
                if layer_idx < length(obj.rbm_layers)
                    state{layer_idx+1}.v_field = state{layer_idx}.h_field;
                    state{layer_idx+1}.v_state = state{layer_idx}.h_state;
                end
            end
            
            t_state = state{length(obj.rbm_layers)}.h_state;
        end
        
        function v_field = likelihood(obj,h_state)
            % likelihood ������������Ԫ��ȡֵ������ײ�����Ԫ����ֵ
            v_field = h_state;
            for layer_idx = length(obj.rbm_layers):-1:1
                v_field = obj.rbm_layers{layer_idx}.likelihood(v_field);
            end
        end
        
        function [b_state,state] = likelihood_sample(obj,h_state)
            % likelihood_sample ����������Ԫ��״̬������ײ�������Ԫ��״̬
            state{length(obj.rbm_layers)}.h_field = h_state;
            state{length(obj.rbm_layers)}.h_state = h_state;

            for layer_idx = length(obj.rbm_layers):-1:1
                state{layer_idx}.v_field = obj.rbm_layers{layer_idx}.likelihood(state{layer_idx}.h_state);
                state{layer_idx}.v_state = ML.sample(state{layer_idx}.v_field);
                if layer_idx > 1
                    state{layer_idx-1}.h_field = state{layer_idx}.v_field;
                    state{layer_idx-1}.h_state = state{layer_idx}.v_state;
                end
            end
            
            b_state = state{1}.v_state;
        end
        
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
        
        function l = layer_num(obj)
            l = length(obj.rbm_layers);
        end
    end
end


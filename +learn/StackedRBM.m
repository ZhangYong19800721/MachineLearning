classdef StackedRBM
%Stacked Restricted Boltzmann Machine �ѵ�����������   
%
    properties
        rbm_layers;
    end
    
    methods
        function obj = StackedRBM(configure) 
            % DeepBeliefNet ���캯��
            layer_num = length(configure) - 1;
            for layer_idx = 1:layer_num
                obj.rbm_layers{layer_idx} = learn.RBM(configure(layer_idx),configure(layer_idx+1));
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
        
        function obj = pretrain(obj,minibatchs,parameters)
            %pretrain Ԥѵ��
            % ʹ��CD1�����㷨�����ѵ��Լ��������������RBM��
            K = length(obj.rbm_layers);
            [D,S,M] = size(minibatchs);
            
            for k = 1:K
                obj.rbm_layers{k} = obj.rbm_layers{k}.initialize(minibatchs); % ��ʼ����layer_idx���RBM
                obj.rbm_layers{k} = obj.rbm_layers{k}.pretrain(minibatchs,parameters); % ѵ����layer_idx���RBM
                minibatchs = reshape(minibatchs,obj.rbm_layers{k}.num_visual,[]);
                minibatchs = obj.rbm_layers{k}.posterior(minibatchs);
                minibatchs = reshape(minibatchs,obj.rbm_layers{k}.num_hidden,S,M);
            end
        end
        
        function l = layer_num(obj)
            l = length(obj.rbm_layers);
        end
    end
end


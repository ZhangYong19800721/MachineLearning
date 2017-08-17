classdef StackedRBM
%Stacked Restricted Boltzmann Machine �ѵ�����������   
%
    properties
        rbms;
    end
    
    methods
        function obj = StackedRBM(configure) 
            % DeepBeliefNet ���캯��
            L = length(configure) - 1;
            for l = 1:L
                obj.rbms{l} = learn.RBM(configure(l),configure(l+1));
            end
        end
    end
    
    methods        
        function proba = posterior(obj,v_state)
            % posterior �����Բ���Ԫ��ȡֵ�����㶥��������Ԫ����ֵ
            L = obj.layer_num(); proba = cell(1,L);
            
            proba{1} = v_state;
            for l = 1:L
                proba{l+1} = obj.rbms{l}.posterior(proba{l});
            end
        end
        
        function state = posterior_sample(obj,v_state)
            % posterior_sample �����Բ���Ԫ��״̬�����㶥��������Ԫ��״̬
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
            % likelihood ������������Ԫ��ȡֵ������ײ�����Ԫ����ֵ
            L = obj.layer_num();
            
            proba{L+1} = h_state;
            for l = L:-1:1
                proba{l} = obj.rbms{l}.likelihood(proba{l+1});
            end
        end
        
        function state = likelihood_sample(obj,h_state)
            % likelihood_sample ����������Ԫ��״̬������ײ�������Ԫ��״̬
            L = obj.layer_num();
            
            state{L+1}.proba = h_state;
            state{L+1}.state = h_state;

            for l = L:-1:1
                state{l}.proba = obj.rbms{l}.likelihood(state{l+1}.state);
                state{l}.state = learn.sample(state{l}.proba);
            end
        end
        
        function obj = pretrain(obj,minibatchs,parameters)
            %pretrain Ԥѵ��
            % ʹ��CD1�����㷨�����ѵ��Լ��������������RBM��
            L = obj.layer_num();
            [D,S,M] = size(minibatchs);
            
            for l = 1:L
                obj.rbms{l} = obj.rbms{l}.initialize(minibatchs); % ��ʼ����l���RBM
                obj.rbms{l} = obj.rbms{l}.pretrain(minibatchs,parameters); % ѵ����l���RBM
                minibatchs = reshape(minibatchs,obj.rbms{l}.num_visual,[]);
                minibatchs = obj.rbms{l}.posterior(minibatchs);
                minibatchs = reshape(minibatchs,obj.rbms{l}.num_hidden,S,M);
            end
        end
        
        function l = layer_num(obj)
            l = length(obj.rbms);
        end
        
        function obj = weightsync(obj)
            L = obj.layer_num(); % ����
            for l = 1:L
                obj.rbms{l} = obj.rbms{l}.weightsync();
            end
        end
    end
end


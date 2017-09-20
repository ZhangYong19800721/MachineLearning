classdef RBM
    %RestrictedBoltzmannMachine Լ������������
    %   
    
    properties
        num_hidden; % ������Ԫ�ĸ���
        num_visual; % �ɼ���Ԫ�ĸ���
        weight_v2h;  % Ȩֵ����(num_hidden * num_visual)
        weight_h2v;  % Ȩֵ����(num_hidden * num_visual)
        hidden_bias; % ������Ԫ��ƫ��
        visual_bias; % �ɼ���Ԫ��ƫ��
        points; % ѵ������
    end
    
    methods
        function obj = RBM(num_visual,num_hidden) % ���캯��
            obj.num_hidden  = num_hidden;
            obj.num_visual  = num_visual;
            obj.weight_v2h  = zeros(obj.num_hidden,obj.num_visual);
            obj.weight_h2v  = [];
            obj.hidden_bias = zeros(obj.num_hidden,1);
            obj.visual_bias = zeros(obj.num_visual,1);
        end
    end
    
    methods
        %% �����ݶ�
        function g = gradient(obj,x,i)
            %% Ƕ�����
            H = obj.num_hidden; V = obj.num_visual; W = H*V;
            obj.weight_v2h (:) = x(     1:W );
            obj.hidden_bias(:) = x(W+  (1:H));
            obj.visual_bias(:) = x(W+H+(1:V));
            
            %%
            [D,S,M] = size(obj.points);
            i = 1 + mod(i,M);
            minibatch = obj.points(:,:,i);
            N = size(minibatch,2); % ѵ�������ĸ���
            h_bias = repmat(obj.hidden_bias,1,N);
            v_bias = repmat(obj.visual_bias,1,N);
            
            %%
            h_field_0 = learn.tools.sigmoid(obj.weight_v2h * minibatch + h_bias);
            h_state_0 = learn.tools.sample(h_field_0);
            v_field_1 = learn.tools.sigmoid(obj.weight_v2h'* h_state_0 + v_bias);
            v_state_1 = v_field_1; 
            h_field_1 = learn.tools.sigmoid(obj.weight_v2h * v_state_1 + h_bias);
            gw = (h_field_0 * minibatch' - h_field_1 * v_state_1') / N;
            gh = (h_field_0 - h_field_1) * ones(N,1) / N;
            gv = (minibatch - v_state_1) * ones(N,1) / N;
            
            weight_cost = 1e-4; cw = weight_cost * obj.weight_v2h;
            g = -[gw(:)-cw(:);gh(:);gv(:)];
        end
        
        %% �����ؽ����
        function y = object(obj,x,i)
            %% Ƕ�����
            H = obj.num_hidden; V = obj.num_visual; W = H*V;
            obj.weight_v2h(:)  = x(     1:W );
            obj.hidden_bias(:) = x(W+  (1:H));
            obj.visual_bias(:) = x(W+H+(1:V));
            
            %%
            [D,S,M] = size(obj.points);
            i = 1 + mod(i,M);
            minibatch = obj.points(:,:,i);
            N = size(minibatch,2); % ѵ�������ĸ���
            h_bias = repmat(obj.hidden_bias,1,N);
            v_bias = repmat(obj.visual_bias,1,N);
            
            %% �����ؽ����
            h_field_0 = learn.tools.sigmoid(obj.weight_v2h * minibatch + h_bias);
            h_state_0 = learn.tools.sample(h_field_0);
            v_field_1 = learn.tools.sigmoid(obj.weight_v2h'* h_state_0 + v_bias);
            y =  sum(sum((v_field_1 - minibatch).^2)) / N; %����������minibatch�ϵ�ƽ���ؽ����
        end
        
        function obj = construct(obj,weight_v2h,weight_h2v,visual_bias,hidden_bias)
            %construct ʹ��Ȩֵ������Ԫƫ�á�����Ԫƫ��ֱ�ӹ���RBM
            [obj.num_hidden, obj.num_visual] = size(weight_v2h);
            obj.weight_v2h = weight_v2h;         
            obj.weight_h2v = weight_h2v;
            obj.hidden_bias = hidden_bias;
            obj.visual_bias = visual_bias;
        end
        
        function obj = initialize(obj,points) 
            %INTIALIZE ��ʼ��Ȩֵ����Ϊ0������С���������ʼ���Բ���Ԫ��ƫ��Ϊ������ʣ���ʼ��������Ԫ��ƫ��Ϊ0.   
            [D,S,M] = size(points);
            points = reshape(points,D,[]);
            obj = obj.initialize_weight(points);
        end

        function obj = pretrain(obj,points,parameters) 
            %pretrain ��Ȩֵ����Ԥѵ��
            % ʹ��CD1�����㷨��Ȩֵ����Ԥѵ��
            
            obj.points = points; % ��ѵ������
            x0 = [obj.weight_v2h(:); obj.hidden_bias(:); obj.visual_bias(:)]; % �趨������ʼ��
            x = learn.optimal.minimize_sgd(obj,x0,parameters); % ����ݶ��½�
            
            %% Ƕ�����
            H = obj.num_hidden; V = obj.num_visual; W = H*V;
            obj.weight_v2h(:)  = x(     1:W );
            obj.hidden_bias(:) = x(W+  (1:H));
            obj.visual_bias(:) = x(W+H+(1:V));
            
            obj.points = []; % �����ѵ������
        end
        
        function h_state = posterior_sample(obj,v_state)
            % posterior_sample ���������ʲ���
            % �ڸ����Բ���Ԫȡֵ������£�������Ԫ���г���
            h_state = learn.tools.sample(obj.posterior(v_state));
        end
        
        function h_field = posterior(obj,v_state) 
            %POSTERIOR ����������
            % �ڸ����Բ���Ԫȡֵ������£���������Ԫ�ļ������
            h_field = learn.tools.sigmoid(obj.foreward(v_state));
        end
        
        function v_state = likelihood_sample(obj,h_state) 
            % likelihood_sample ������Ȼ���ʲ���
            % �ڸ���������Ԫȡֵ������£�������Ԫ���г���
            v_state = learn.sample(obj.likelihood(h_state));
        end
        
        function v_field = likelihood(obj,h_state) 
            % likelihood ������Ȼ����
            % �ڸ���������Ԫȡֵ������£���������Ԫ�ļ������
            v_field = learn.tools.sigmoid(obj.backward(h_state));
        end
        
        function y = foreward(obj,x)
            y = obj.weight_v2h * x + repmat(obj.hidden_bias,1,size(x,2));
        end
        
        function x = backward(obj,y)
            if isempty(obj.weight_h2v)
                x = obj.weight_v2h'* y + repmat(obj.visual_bias,1,size(y,2));
            else
                x = obj.weight_h2v'* y + repmat(obj.visual_bias,1,size(y,2));
            end
        end
        
        function y = rebuild(obj,x)
            z = obj.posterior_sample(x);
            y = obj.likelihood(z);
        end
        
        function obj = weightsync(obj)
            obj.weight_h2v = obj.weight_v2h;
        end
    end
    
    methods (Access = private)
        function obj = initialize_weight(obj,points)
            %INTIALIZE ��ʼ��Ȩֵ����Ϊ0������С���������ʼ���Բ���Ԫ��ƫ��Ϊ������ʣ���ʼ��������Ԫ��ƫ��Ϊ0.
            obj.weight_v2h = 0.01 * randn(size(obj.weight_v2h));
            obj.visual_bias = mean(points,2);
            obj.visual_bias = log(obj.visual_bias./(1-obj.visual_bias));
            obj.visual_bias(obj.visual_bias < -100) = -100;
            obj.visual_bias(obj.visual_bias > +100) = +100;
            obj.hidden_bias = zeros(size(obj.hidden_bias));
        end
    end
    
    methods(Static)
        function [] = unit_test()
            clear all;
            close all;
            rng(1);
            
            [data,~,~,~] = learn.data.import_mnist('./+learn/+data/mnist.mat');
            [D,S,M] = size(data); N = S * M;
     
            rbm = learn.neural.RBM(D,500);
            rbm = rbm.initialize(data);
            
            parameters.learn_rate = 1e-1;
            parameters.max_it = M*100; % �����е�ѵ�����ݱ���X��
            rbm = rbm.pretrain(data,parameters);
            
            data = reshape(data,D,[]);
            rebuild_data = rbm.rebuild(data);
            e = sum(sum((rebuild_data - data).^2)) / N;
            disp(sprintf('�ؽ����:%f',e));
        end
    end
    
end


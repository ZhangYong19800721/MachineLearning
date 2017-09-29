classdef PerceptionS
    %PERCEPTIONS ��֪��
    %  ���һ�����Ϊsigmoid��Ԫ�����ý�������ΪĿ�꺯��������ʹ��CG��BFGS�㷨����Ѱ��
    
    properties
        weight;      % һά���飬���в��Ȩֵ��ƫ��ֵ�����������һά������[P,1]
        num_hidden;  % num_hidden{m}��ʾ��m�������Ԫ����
        num_visual;  % num_visual{m}��ʾ��m�������Ԫ����
        star_w_idx;  % star_w_idx{m}��ʾ��m���Ȩֵ����ʼλ��
        stop_w_idx;  % stop_w_idx{m}��ʾ��m���Ȩֵ�Ľ���λ��
        star_b_idx;  % star_b_idx{m}��ʾ��m���ƫ�õ���ʼλ��
        stop_b_idx;  % stop_b_idx{m}��ʾ��m���ƫ�õĽ���λ��
        points; % ѵ������
        labels; % ѵ����ǩ
    end
    
    methods % ���캯��
        function obj = PerceptionS(configure)
            % Perception ���캯��
            M = length(configure) - 1;
            obj.num_hidden{1} = configure(2);
            obj.num_visual{1} = configure(1);
            obj.star_w_idx{1} = 1;
            obj.stop_w_idx{1} = obj.num_hidden{1} * obj.num_visual{1};
            obj.star_b_idx{1} = obj.stop_w_idx{1} + 1;
            obj.stop_b_idx{1} = obj.stop_w_idx{1} + obj.num_hidden{1};
            
            for m = 2:M
                obj.num_hidden{m} = configure(m+1);
                obj.num_visual{m} = configure(m+0);
                obj.star_w_idx{m} = obj.stop_b_idx{m-1} + 1;
                obj.stop_w_idx{m} = obj.stop_b_idx{m-1} + obj.num_hidden{m} * obj.num_visual{m};
                obj.star_b_idx{m} = obj.stop_w_idx{m} + 1;
                obj.stop_b_idx{m} = obj.stop_w_idx{m} + obj.num_hidden{m};
            end
        end
    end
    
    methods
        %% �ݶȼ���
        function g = gradient(obj,x,m)
            %% ��ʼ��
            [D,S,M] = size(obj.points); % D����ά�ȣ�S����������M��������
            P = size(obj.weight,1); % P��������
            L = length(obj.num_hidden); % L����
            g = zeros(P,1); % �ݶ�
            
            if nargin <= 2 % û�и���i������Ĭ�϶�ȫ��ѵ����������
                for m = 1:M
                    g = g + obj.gradient(x,m);
                end
            else
                m = 1 + mod(m,M);
                minibatch = obj.points(:,:,m); % ȡһ��minibatch
                minilabel = obj.labels(:,:,m); % ȡһ��minibatch
                obj.weight = x; % ��ʼ��Ȩֵ
                s = cell(1,L); % ������
                w = cell(1,L); iw = cell(1,L); % Ȩֵ
                b = cell(1,L); ib = cell(1,L); % ƫ��
                for l = 1:L % ȡ��ÿһ���Ȩֵ��ƫ��ֵ
                    [w{l},iw{l}] = obj.getw(l);
                    [b{l},ib{l}] = obj.getb(l);
                end
                
                %% �����ݶ�
                [y,a] = obj.do(minibatch); % ִ���������
                s{L} = (y - minilabel)'; % ���㶥���������
                for l = (L-1):-1:1  % ���򴫲�������
                    s{l} = (s{l+1} * w{l+1}) .* (a{l}.*(1-a{l}))';
                end
                
                for l = 1:L
                    H = obj.num_hidden{l};
                    V = obj.num_visual{l};
                    if l == 1
                        gx = reshape(repmat(s{l}',V,1),H,V,S) .* reshape(repelem(minibatch,H,1),H,V,S);
                    else
                        gx = reshape(repmat(s{l}',V,1),H,V,S) .* reshape(repelem(a{l-1}   ,H,1),H,V,S);
                    end
                    gx = sum(gx,3);
                    g(iw{l},1) = g(iw{l},1) + gx(:);
                    g(ib{l},1) = g(ib{l},1) + sum(s{l},1)';
                end
            end
        end
        
        %% ����Ŀ�꺯��
        function y = object(obj,x,m)
            %% ��ʼ��
            [D,S,M] = size(obj.points);
            
            if nargin <= 2 % û�и���m������Ĭ�϶�ȫ��ѵ����������
                y = 0;
                for m = 1:M
                    y = y + obj.object(x,m);
                end
            else
                obj.weight = x; % ��ʼ��Ȩֵ
                m = 1 + mod(m,M);
                minibatch = obj.points(:,:,m); % ȡһ��minibatch
                minilabel = obj.labels(:,:,m); % ȡһ��minibatch
                z = obj.compute(minibatch);
                z(z<=0) = eps; z(z>=1) = 1 - eps;
                y = minilabel .* log(z) + (1-minilabel) .* log(1-z); % ���㽻����
                y = -sum(sum(y));
            end
        end
        
        function obj = initialize(obj)
            M = length(obj.num_hidden); % �õ�����
            for m = 1:M
                obj.weight(obj.star_w_idx{m}:obj.stop_w_idx{m},1) = ...
                    0.01 * randn(size([obj.star_w_idx{m}:obj.stop_w_idx{m}]')); % ��Ȩֵ��ʼ��Ϊ0�����������
                obj.weight(obj.star_b_idx{m}:obj.stop_b_idx{m},1) = ...
                    zeros(size([obj.star_b_idx{m}:obj.stop_b_idx{m}]')); % ��ƫ��ֵ��ʼ��Ϊ0
            end
        end
		
		function y = compute(obj,x,L)
            % ����֪���ļ������
            % y ���������
            
            if nargin <= 2
                L = length(obj.num_hidden); % �õ�����
            end
            
            for l = 1:L
                w = obj.getw(l);
                b = obj.getb(l);
                y = learn.tools.sigmoid(w * x + repmat(b,1,size(x,2)));
                x = y;
            end
        end
        
        function [y,a] = do(obj,x,L)
            % ����֪���ļ������
            % y ���������
            
            if nargin <= 2
                L = length(obj.num_hidden); % �õ�����
            end
            
            a = cell(1,L);
            for l = 1:L
                w = obj.getw(l);
                b = obj.getb(l);
                a{l} = learn.tools.sigmoid(w * x + repmat(b,1,size(x,2)));
                x = a{l};
            end
            
            y = a{L};
        end
        
        function [w,r] = getw(obj,m)
            r = obj.star_w_idx{m}:obj.stop_w_idx{m};
            w = reshape(obj.weight(r),obj.num_hidden{m},obj.num_visual{m});
        end
        
        function [b,r] = getb(obj,m)
            r = obj.star_b_idx{m}:obj.stop_b_idx{m};
            b = reshape(obj.weight(r),[],1);
        end
        
        function obj = train(obj,points,labels,parameters)
            if nargin <= 3
                parameters = [];
                disp('����train����ʱû�и�������������ʹ��Ĭ�ϲ�����');
            end
            
            if ~isfield(parameters,'algorithm')
                parameters.algorithm = 'CG';
                disp(sprintf('û��algorithm��������ʹ��Ĭ��ֵ%s',parameters.algorithm));
            end
            
            %% ��ѵ������
            obj.points = points;
            obj.labels = labels;
            
            %% Ѱ��
            if strcmp(parameters.algorithm,'CG')
                obj.weight = learn.optimal.minimize_cg(obj,obj.weight,parameters);
            elseif strcmp(parameters.algorithm,'BFGS')
                obj.weight = learn.optimal.minimize_bfgs(obj,obj.weight,parameters);
            elseif strcmp(parameters.algorithm,'LM')
                obj.weight = learn.optimal.minimize_lm(obj,obj.weight,parameters);
			elseif strcmp(parameters.algorithm,'ADAM')
                obj.weight = learn.optimal.minimize_adam(obj,obj.weight,parameters);
			elseif strcmp(parameters.algorithm,'SGD')
                obj.weight = learn.optimal.minimize_sgd(obj,obj.weight,parameters);
			elseif strcmp(parameters.algorithm,'GD')
                obj.weight = learn.optimal.minimize_g(obj,obj.weight,parameters);
            end
            
            %% �����
            obj.points = [];
            obj.labels = [];
        end
    end
    
    methods(Static)
        function [] = unit_test1()
            clear all;
            close all;
            rng(1);
            
            N = 2000;
            x = linspace(-2,2,N);
            k = 2;
            f = @(x)0.5 + 0.5 * sin(k * pi * x / 4);
            l = f(x);
            
            configure = [1,12,1];
            p = learn.neural.PerceptionS(configure);
            p = p.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,p.do(x)); hold off;
            
            paramters.algorithm = 'BFGS';
            % paramters.algorithm = 'CG';
            paramters.epsilon = 1e-3;
            paramters.max_it = 1e6;
            p = p.train(x,l,paramters);
            
            figure(3);
            y = p.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            disp(sprintf('error:%f',sum(sum((l - y).^2))));
        end
        
        function [] = unit_test2()
            clear all;
            close all;
            rng(1);
            
            x = [0.8 0.2];
            l = [0.2 0.8];
            
            configure = [1,2,1];
            p = learn.neural.PerceptionS(configure);
            p = p.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,p.do(x)); hold off;
            
            parameters.epsilon = 1e-8;
            p = p.train(x,l,parameters);
            
            figure(3);
            y = p.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            disp(sprintf('error:%f',sum(sum((l - y).^2))));
        end
    end
end
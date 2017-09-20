classdef PerceptionL
    %PERCEPTIONL ��֪��
    % ���һ�����Ϊ������Ԫ
    
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
        function obj = PerceptionL(configure)
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
        %% �����ݶ�
        function g = gradient(obj,x,i)
            %% ��ʼ��
            obj.weight = x; % ��ʼ��Ȩֵ
            [D,S,M] = size(obj.points);
            i = 1 + mod(i,M);
            minibatch = obj.points(:,:,i); % ȡһ��minibatch
            minilabel = obj.labels(:,:,i); % ȡһ��minibatch
            N = size(minibatch,2); % N��������
            P = size(obj.weight,1); % P��������
            M = length(obj.num_hidden); % M����
            g = zeros(P,1); % �ݶ�
            s = cell(1,M); % ������
            w = cell(1,M); % Ȩֵ
            b = cell(1,M); % ƫ��
            for m = 1:M % ȡ��ÿһ���Ȩֵ��ƫ��ֵ
                w{m} = obj.getw(m);
                b{m} = obj.getb(m);
            end
            
            %% �����ݶ�
            [~,a] = obj.do(minibatch); % ִ���������
            s{M} = -2 * (minilabel - a{M})'; % ���㶥���������
            for m = (M-1):-1:1  % ���򴫲�������
                sx = s{m+1}; wx = w{m+1}; ax = a{m}.*(1-a{m});
                sm = zeros(N,obj.num_hidden{m});
                parfor n = 1:N
                    sm(n,:) = sx(n,:) * wx * diag(ax(:,n));
                end
                s{m} = sm;
            end
                
            for m = 1:M
                [~,cw] = obj.getw(m);
                [~,cb] = obj.getb(m);

                H = obj.num_hidden{m};
                V = obj.num_visual{m};
                
                sx = s{m}'; 
                if m == 1
                    px = minibatch';
                    gx = zeros(size(w{m}));
                    parfor n = 1:N
                        gx = gx + repmat(sx(:,n),1,V) .* repmat(px(n,:),H,1);
                    end
                else
                    ax = a{m-1}';
                    gx = zeros(size(w{m}));
                    parfor n = 1:N
                        gx = gx + repmat(sx(:,n),1,V) .* repmat(ax(n,:),H,1);
                    end
                end

                g(cw,1) = g(cw,1) + gx(:);
                g(cb,1) = g(cb,1) + sum(sx,2);
            end
            
            g = g ./ N;
        end
        
        %% ���㺯��ֵ
        function y = object(obj,x,i)
           %% ��ʼ��
            obj.weight = x; % ��ʼ��Ȩֵ
            [D,S,M] = size(obj.points);
            i = 1 + mod(i,M);
            minibatch = obj.points(:,:,i); % ȡһ��minibatch
            minilabel = obj.labels(:,:,i); % ȡһ��minibatch
            
           %% ����Ŀ�꺯��
            z = obj.do(minibatch);
            y = sum(sum((minilabel - z).^2));
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
        
        function [y,a] = do(obj,x,M)
            % ����֪���ļ������
            % y ���������
            
            if nargin <= 2
                M = length(obj.num_hidden); % �õ�����
            end
           
            a = cell(1,M);          
            for m = 1:M
                w = reshape(obj.weight(obj.star_w_idx{m}:obj.stop_w_idx{m}),obj.num_hidden{m},obj.num_visual{m});
                b = reshape(obj.weight(obj.star_b_idx{m}:obj.stop_b_idx{m}),obj.num_hidden{m},1);
                n = w * x + repmat(b,1,size(x,2));
                if m < M
                    a{m} = learn.tools.sigmoid(n);
                    x = a{m};
                else
                    a{m} = n;
                    y = n;
                end
            end
        end
        
        function [w,r] = getw(obj,m)
            r = obj.star_w_idx{m}:obj.stop_w_idx{m};
            w = reshape(obj.weight(r),obj.num_hidden{m},obj.num_visual{m});
        end
        
        function [b,r] = getb(obj,m)
            r = obj.star_b_idx{m}:obj.stop_b_idx{m};
            b = reshape(obj.weight(r),[],1);
        end
    end
    
    methods(Static)
        function p = unit_test1()
            clear all;
            close all;
            rng(1);
            
            N = 2000;
            x = linspace(-2,2,N);
            k = 4;
            f = @(x)sin(k * pi * x / 4);
            l = f(x);
            
            configure = [1,6,1];
            p = learn.neural.PerceptionL(configure);
            p = p.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,p.do(x)); hold off;
            
            lmbp = learn.neural.LMBPL(x,l,p);
            weight = learn.optimal.minimize_lm(lmbp,p.weight);
            p.weight = weight;
            
            figure(3);
            y = p.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            disp(sprintf('error:%f',sum(sum((l - y).^2))));
        end
        
        function p = unit_test2()
            clear all;
            close all;
            rng(1);
            
            N = 2000;
            x = linspace(-2,2,N);
            k = 4;
            f = @(x)sin(k * pi * x / 4);
            l = f(x);
            
            configure = [1,6,1];
            p = learn.neural.PerceptionL(configure);
            p = p.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,p.do(x)); hold off;
            
            cgbp = learn.neural.CGBPL(x,l,p);
            weight = learn.optimal.minimize_cg(cgbp,p.weight);
            p.weight = weight;
            
            figure(3);
            y = p.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            disp(sprintf('error:%f',sum(sum((l - y).^2))));
        end
        
        function p = unit_test3()
            clear all;
            close all;
            rng(1);
            
            N = 2000;
            x = linspace(-2,2,N);
            k = 4;
            f = @(x)sin(k * pi * x / 4);
            l = f(x);
            
            configure = [1,6,1];
            p = learn.neural.PerceptionL(configure);
            p = p.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,p.do(x)); hold off;
            
            cgbp = learn.neural.CGBPL(x,l,p);
            weight = learn.optimal.minimize_bfgs(cgbp,p.weight);
            p.weight = weight;
            
            figure(3);
            y = p.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            disp(sprintf('error:%f',sum(sum((l - y).^2))));
        end
        
        function [] = unit_test4()
            clear all;
            close all;
            rng(1);
            
            N = 2000;
            x = linspace(-2,2,N);
            k = 4;
            f = @(x)sin(k * pi * x / 4);
            l = f(x);
            
            configure = [1,6,1];
            p = learn.neural.PerceptionL(configure);
            p = p.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,p.do(x)); hold off;
            
            
            p.points = reshape(x,1,20,100);
            p.labels = reshape(l,1,20,100);
            weight = learn.optimal.minimize_adam(p,p.weight);
            p.weight = weight;
            
            figure(3);
            y = p.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            disp(sprintf('error:%f',sum(sum((l - y).^2))));
        end
    end
end
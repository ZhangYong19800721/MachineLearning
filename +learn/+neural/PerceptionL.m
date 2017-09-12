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
        function obj = initialize(obj)
            M = length(obj.num_hidden); % �õ�����
            for m = 1:M
                obj.weight(obj.star_w_idx{m}:obj.stop_w_idx{m},1) = ...
                    0.01 * randn(size([obj.star_w_idx{m}:obj.stop_w_idx{m}]')); % ��Ȩֵ��ʼ��Ϊ0�����������
                obj.weight(obj.star_b_idx{m}:obj.stop_b_idx{m},1) = ...
                    zeros(size([obj.star_b_idx{m}:obj.stop_b_idx{m}]')); % ��ƫ��ֵ��ʼ��Ϊ0
            end
        end
        
        function [y,a] = do(obj,x)
            % ����֪���ļ������
            % y ���������
            
            M = length(obj.num_hidden); % �õ�����
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
        function [perception,e] = unit_test1()
            clear all;
            close all;
            rng(1);
            
            N = 2000;
            x = linspace(-2,2,N);
            k = 3;
            f = @(x)sin(k * pi * x / 4);
            l = f(x);
            
            configure = [1,6,1];
            perception = learn.Perception(configure);
            perception = perception.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,perception.do(x)); hold off;
            
            lmbp = learn.LevenbergMarquardtBP(x,l,perception);
            
            weight = optimal.LevenbergMarquardt(lmbp,perception.weight,1e-6,1e5);
            perception.weight = weight;
            
            figure(3);
            y = perception.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            e = norm(l - y,2);
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
            
            parameters.epsilon = 1e-5;
            parameters.alfa = 1000;
            parameters.beda = 1e-8;
            parameters.max_it = 1e5;
            parameters.reset = 500;
            weight = learn.optimal.minimize_cg(cgbp,p.weight,parameters);
            p.weight = weight;
            
            figure(3);
            y = p.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            disp(sprintf('error:%f',norm(l - y,2)));
        end
    end
end
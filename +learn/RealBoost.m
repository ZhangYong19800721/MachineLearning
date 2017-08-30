classdef RealBoost
    %RealBoost ʵ��Real Ada Boost�㷨
    %  �ο�����"Additve Logistic Regression a Statistical View of Boosting"
    %  ��ƪ���ķǳ���Ҫ����ʾ��boost�㷨�ͼ���ͳ��ģ���ڱ���������ͬ�ģ�����
    %  ��RealAdaBoost��GentleBoost�㷨��
    
    properties
        weak; % �������ɸ�����������cell����
    end
    
    methods
        function obj = RealBoost()
        end
    end
    
    methods(Access = private)
        function [obj,w]  = ensemble(obj,points,labels,weight,wc)
            %ensemble ��������������ҵ�һ�����ŵ�����������Ȼ��ͨ���˺����������
            %   ���룺
            %   points ��ʾ���ݵ�
            %   labels ��ǩ+1��-1
            %   weight ���ʷֲ�Ȩֵ
            %   wc �¼������������
            % 
            %   �����
            %   obj ѵ�����adaboost����
            %   w ���º��Ȩֵ
            
            f = wc.compute(points); % ����������������ֵf(x)
            obj.weak{1+length(obj.weak)} = wc; % ������������������
            w = weight .* exp(-labels .* f); % ����Ȩֵ
            w = w ./ sum(w); % ��һ��Ȩֵ
        end
        
        function wc = find_wc(obj,points,labels,weight)
            % ѡ�����ŵ���������
            wc = learn.Weak_LineR(); l = labels > 0;
            [~,N] = size(points); epsilon = 1/N;
            
            X = -15:15; Y = -15:15; A = linspace(-pi/2,pi/2,180); 
            zm = inf;
            for x = X 
                for y = Y
                    for a = A
                        v = [-tan(a) 1 tan(a)*x-y];
                        wc.w = [v(1) v(2)]; wc.b = v(3);
                        p = wc.predict(points);
                        W_POS_1 = sum(weight( l &  p));
                        W_NEG_1 = sum(weight(~l &  p));
                        W_POS_2 = sum(weight( l & ~p));
                        W_NEG_2 = sum(weight(~l & ~p));
                        z = 2 * sum(sqrt(W_POS_1 * W_NEG_1) + sqrt(W_POS_2 * W_NEG_2));
                        if z < zm
                            best.w = wc.w;
                            best.b = wc.b;
                            best.x = 0.5 * log((W_POS_1 + epsilon) / (W_NEG_1 + epsilon));
                            best.y = 0.5 * log((W_POS_2 + epsilon) / (W_NEG_2 + epsilon));
                            zm = z;
                        end
                    end
                end
            end
            
            wc.w = best.w;
            wc.b = best.b;
            wc.x = best.x;
            wc.y = best.y;
        end
    end
    
    methods(Access = public)
        function y = compute(obj,points)
            % compute ����boostģ�͵���Ӧ����F(x)
            M = length(obj.weak); [~,N] = size(points); % M���������ĸ�����N���ݵ���
            f = zeros(M,N);                             % �洢���������ļ�����
            for m=1:M,f(m,:) = obj.weak{m}.compute(points);end
            y = sum(f,1);
        end
        
        function y = predict(obj,points)  
            % PREDICT ʹ�þ���ѵ����ģ���ж����ݵ�ķ��࣬����Ϊ1������Ϊ0
            y = obj.compute(points) > 0;
        end
        
        function obj = train(obj,points,labels,M)
            % train ѵ��RealBoostģ��
            % ���룺
            % points ���ݵ�
            % labels ��ǩ��+1��-1
            % M ���������ĸ���
            % �����
            % obj ����ѵ����boost����
            
            [~,N] = size(points); % �õ����ݵ���Ŀ
            weight = ones(1,N) / N; % ��ʼ��Ȩֵ
            for m = 1:M
                wc = obj.find_wc(points,labels,weight);
                [obj,weight] = obj.ensemble(points,labels,weight,wc);
                
                l = labels > 0;
                y = obj.predict(points);
                error(m) = sum(xor(y,l)) / N
                
                a = obj.weak{m}.w(1);
                b = obj.weak{m}.w(2);
                c = obj.weak{m}.b;
                myfunc = @(x,y) a*x + b*y + c;
                ezplot(myfunc,[-15,15,-8,8]);
                drawnow
            end
        end
    end
    
    methods(Static)
        function boost = unit_test()
            clear all;
            close all;
            rng(1)
            
            boost = learn.RealBoost();
            
            N = 1e3;
            [points,labels] = learn.GenerateData.type1(N); l = labels > 0;
            
            figure;
            group1 = points(:,labels== 1);
            group2 = points(:,labels==-1);
            plot(group1(1,:),group1(2,:),'+'); hold on;
            plot(group2(1,:),group2(2,:),'*'); 
            
            M = 15;
            boost = boost.train(points,labels,M);
            
            y = boost.predict(points);
            error = sum(xor(y,l)) / N
        end
    end
end


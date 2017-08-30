classdef GentleBoost
    %GENTLEBOOST GentleBoost�㷨
    %  �ο�����"Additive Logistic Regression a Statistical View of Boosting"
    %  ��ƪ���ķǳ���Ҫ����ʾ��boost�㷨�ͼ���ͳ��ģ���ڱ���������ͬ�ģ�����
    %  ��AdaBoost��RealBoost��LogitBoost��GentleBoost�㷨��
    
    properties
        weak; % �������ɸ�����������cell����
    end
    
    methods(Access = private)
        function [obj,w]  = ensemble(obj,points,labels,weight,wc)
            %ensemble �����������
            %   ���룺
            %   points ��ʾ���ݵ�
            %   labels ��ǩ1��0
            %   weight ���ʷֲ�Ȩֵ
            %   wc �¼������������
            % 
            %   �����
            %   obj ѵ�����adaboost����
            %   w ���º��Ȩֵ
            
            f = wc.compute(points); % ���ȼ����������������
            obj.weak{1+length(obj.weak)} = wc; % �������������뵽boost����
            w = weight .* exp(-labels .* f); % ����Ȩֵ
            w = w ./ sum(w); % ��һ��Ȩֵ
        end
        
        function wc = find_wc(obj,points,labels,weight)
            % ѡ�����ŵ���������
            wc = learn.Weak_LineR(); l = labels > 0; epsilon = 1e-100;
            
            X = -10:10; Y = -10:10; A = linspace(-pi/2,pi/2,180); 
            zm = inf;
            for x = X 
                for y = Y
                    for a = A
                        v = [-tan(a) 1 tan(a)*x-y];
                        wc.w = [v(1) v(2)]; wc.b = v(3);
                        b = wc.predict(points);
                        W_POS_1 = sum(weight( l &  b)) / (epsilon+sum(weight( b)));
                        W_NEG_1 = sum(weight(~l &  b)) / (epsilon+sum(weight( b)));
                        W_POS_2 = sum(weight( l & ~b)) / (epsilon+sum(weight(~b)));
                        W_NEG_2 = sum(weight(~l & ~b)) / (epsilon+sum(weight(~b)));
                        k = ones(size(b));
                        k( b) = W_POS_1 - W_NEG_1; 
                        k(~b) = W_POS_2 - W_NEG_2;
                        z = weight * ((labels - k).^2)';
                        if z < zm
                            best.w = wc.w;
                            best.b = wc.b;
                            best.x = W_POS_1 - W_NEG_1;
                            best.y = W_POS_2 - W_NEG_2;
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
            M = length(obj.weak); [~,N] = size(points); % M���������ĸ�����N���ݵ���
            f = zeros(M,N); % �洢���������ķ�����
            for m=1:M, f(m,:) = obj.weak{m}.compute(points); end
            y = sum(f,1);
        end
        
        function y = predict(obj,points)  % ʹ�þ���ѵ����ģ���ж����ݵ�ķ���
            %PREDICT �����ݵ���з��࣬����Ϊ1������Ϊ0
            %
            y = obj.compute(points) > 0;
        end
        
        function [obj,err] = train(obj,points,labels,M)
            % train ѵ��GentleBoostģ��
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
                err(m) = sum(xor(y,l)) / N
                
                a = obj.weak{m}.w(1);
                b = obj.weak{m}.w(2);
                c = obj.weak{m}.b;
                myfunc = @(x,y) a*x + b*y + c;
                ezplot(myfunc,[-10,10,-10,10]);
                drawnow;
            end
        end
    end
    
    methods(Static)
        function [boost,err] = unit_test()
            clear all;
            close all;
            rng(2)
            
            boost = learn.GentleBoost();
            
            N = 1e4;
            [points,labels] = learn.GenerateData.type4(N); l = labels > 0;
            
            figure;
            group1 = points(:,labels== 1);
            group2 = points(:,labels==-1);
            plot(group1(1,:),group1(2,:),'+'); hold on;
            plot(group2(1,:),group2(2,:),'.'); 
            
            M = 200;
            [boost,err] = boost.train(points,labels,M);
        end
    end
end


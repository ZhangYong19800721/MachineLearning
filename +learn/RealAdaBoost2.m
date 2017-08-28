classdef RealAdaBoost2
    %RealAdaBoost ʵ��Real AdaBoost�㷨
    %  �ο�����"Additve Logistic Regression a Statistical View of Boosting"
    %  ��ƪ���ķǳ���Ҫ����ʾ��boost�㷨�ͼ���ͳ��ģ���ڱ���������ͬ�ģ�����
    %  ��RealAdaBoost��GentleBoost�㷨��
    
    properties
        weak; % �������ɸ�����������cell����
    end
    
    methods
        function obj = RealAdaBoost2()
        end
    end
    
    methods(Access = public)
        function y = predict(obj,points)  
            % PREDICT ʹ�þ���ѵ����ģ���ж����ݵ�ķ��࣬����Ϊ1������Ϊ0
            %
            H = length(obj.weak); [~,N] = size(points); % H���������ĸ�����N���ݵ���
            c = zeros(H,N);                             % �洢���������ķ�����
            for h=1:H,c(h,:) = obj.weak{h}.predict(points);end
            y = sum(c) > 0;
        end
        
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
            
            f = wc.predict(points); % f��һ��������ʵ�����ϵı���
            obj.weak{1+length(obj.weak)} = wc;
            w = weight .* exp(-labels .* f);
            w = w ./ sum(w);
        end
    end
    
    methods(Static)
        function boost = unit_test()
            clear all;
            close all;
            rng(1)
            
            boost = learn.RealAdaBoost2();
            
            N = 1e4;
            [points,labels] = learn.GenerateData.type1(N); l = labels > 0;
            
            figure;
            group1 = points(:,labels== 1);
            group2 = points(:,labels==-1);
            plot(group1(1,:),group1(2,:),'+'); hold on;
            plot(group2(1,:),group2(2,:),'*'); 
            
            weight = ones(1,N)/N; % ��ʼ��Ȩֵ
            [X,Y] = meshgrid(-15:15,-15:15); A = (-pi/2+eps):0.1:(pi/2-eps); 
            X = reshape(X,1,[]); Y = reshape(Y,1,[]);
            T = 10;
            
            for t = 1:T
                z_min = inf; best_w = []; best_b = [];
                wc = learn.Weak_LineS();
                for i = 1:length(X)
                    for j = 1:length(A)
                        v = [tan(A(j)) 1 -(tan(A(j))*X(i)+Y(i))];
                        wc.w = [v(1) v(2)]; wc.b = v(3);
                        f = wc.predict(points); 
                        z = weight * exp(-labels .* f)';
                        if z < z_min
                            best_w = wc.w;
                            best_b = wc.b;
                            z_min = z;
                        end
                    end
                end
                
                wc.w = best_w; wc.b = best_b;
                [boost,weight] = boost.ensemble(points,labels,weight,wc);
            end
            
            for t = 1:T
                a = boost.weak{t}.w(1);
                b = boost.weak{t}.w(2);
                c = boost.weak{t}.b;
                myfunc = @(x,y) a*x + b*y + c;
                ezplot(myfunc,[-15,15,-8,8]);
            end
            
            c = boost.predict(points);
            error = sum(xor(c,l)) / N;
        end
    end
end


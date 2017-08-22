classdef DiscreteAdaBoost2
    %DiscreteAdaBoost2 ʵ��Discrete AdaBoost2�㷨
    %  �ο�����"Additve Logistic Regression a Statistical View of Boosting"
    %  ��ƪ���ķǳ���Ҫ����ʾ��boost�㷨�ͼ���ͳ��ģ���ڱ���������ͬ�ģ�����
    %  ��RealAdaBoost��GentleBoost�㷨��
    %  DiscreteAdaBoost2��DiscreteAdaBoost1�ǵȼ۵�
    
    properties
        weak; % �������ɸ�����������cell����
        alfa; % ÿ������������ͶƱȨֵ
    end
    
    methods
        function obj = DiscreteAdaBoost2()
        end
    end
    
    methods
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
            
            c = wc.predict(points); % c��һ��logical����
            l = labels > 0; % labelsҲ����logical����
            r = weight * xor(c,l)';
            beda = log((1 - r)/r);
            obj.weak{1+length(obj.weak)} = wc;
            obj.alfa = [obj.alfa beda];
            w = weight .* exp(beda .* xor(c,l));
            w = w ./ sum(w);
        end
        
        function y = predict(obj,points)  % ʹ�þ���ѵ����ģ���ж����ݵ�ķ���
            %PREDICT �ж��������Ƿ����ƣ�����Ϊ+1��������Ϊ-1
            %
            H = length(obj.alfa); [~,N] = size(points); % H���������ĸ�����N���ݵ���
            c = false(H,N); % �洢���������ķ�����
            for h=1:H,c(h,:) = obj.weak{h}.predict(points);end
            k = -ones(size(c)); k(c) = 1;
            y = obj.alfa * k > 0;
        end
    end
    
    methods(Static)
        function boost = unit_test()
            clear all;
            close all;
            rng(1)
            
            boost = learn.DiscreteAdaBoost2();
            
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
                r_min = inf; best_w = []; best_b = [];
                wc = learn.Weak_Line();
                for i = 1:length(X)
                    for j = 1:length(A)
                        wc.w = [tan(A(j)) 1]; wc.b = -(tan(A(j))*X(i)+Y(i));
                        c = wc.predict(points);
                        r = weight * xor(c,l)';
                        if r < r_min
                            best_w = wc.w;
                            best_b = wc.b;
                            r_min = r;
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


classdef DiscreteAdaBoost
    %DiscreteAdaBoost DiscreteAdaBoost�㷨ʵ��
    %  �ο�����"Additive Logistic Regression a Statistical View of Boosting"
    %  ��ƪ���ķǳ���Ҫ����ʾ��boost�㷨�ͼ���ͳ��ģ���ڱ���������ͬ�ģ�����
    %  ��AdaBoost��RealAdaBoost��LogitBoost��GentleAdaBoost�㷨��
    
    properties
        weak; % �������ɸ�����������cell����
        alfa; % ÿ������������ͶƱȨֵ
    end
    
    methods(Access = private)
         %% ѡ��������������
        function [t,a,b,z] = select_stump(obj,points,labels,weight)
            %select_t �ڸ���ά�ȵ������ѡ�����ŵ�����ֵ
            % ��������fm��һ��stump��������4������ȷ����(a,b,k,t)
            % ����DiscreteAdaBoost(a=+2��b=-1)��(a=-2,b=+1)����stump�������ص�ֵֻ����+1��-1
            % fm = a * (x(k) > t) + b
            % ���룺
            %   points ���ݵ㣨ֻ������kά�����ݣ�ֻ��һ�У�
            %   labels ��ǩ,+1��-1
            %   weight Ȩֵ,��Ϊ1�ķǸ�����
            % �����
            %   t ����
            %   a+b 
            %   b
            %   z ���������Ż���Ŀ�꺯��z=sum(weight.*labels.*fm)
            
            %% ��ʼ��
            [K,N] = size(points); % K���ݵ�ά�ȣ�N���ݵ���
            
            %% �������п��ܵ�����ֵT 
            T = unique(points); % ȥ���ظ�ֵ������
            T = [T(1)-eps, (T(1:(end-1)) + T(2:end))/2, T(end)+eps]; % �õ����п��ܵ�����ֵ
            
            %% �����п��ܵ�����ֵ����a,b,z
            A = zeros(size(T)); B = zeros(size(T)); Z = zeros(size(T));
            for m = 1:length(T)
                t = T(m); 
                p =  2 * (points > t) - 1; 
                q = -2 * (points > t) + 1;
                Z1 = sum(weight.*labels.*p);
                Z2 = sum(weight.*labels.*q);
                if Z1 > Z2
                    Z(m) = Z1; A(m) =  2; B(m) = -1;
                else
                    Z(m) = Z2; A(m) = -2; B(m) =  1;
                end
            end
            
            %% ��������������ŵ�����
            [z, best.i] = max(Z);
            t = T(best.i);
            a = A(best.i);
            b = B(best.i);
        end
        
        %% ѡ��������������
        function wc = select_wc(obj,points,labels,weight)
            %select_wc ѡ�����ŵ�stump��������
            % ����������fm��һ��stump��������4������ȷ����(a,b,k,t)
            % fm = a * (x(k) > t) + b
            % ���룺
            %   points ���ݵ�
            %   labels ��ǩ,+1��-1
            %   weight Ȩֵ
            % �����
            %   wc ���ŵ���������
            
            %% ��ʼ��
            [K,N] = size(points); % K ���ݵ�ά�ȣ�N���ݵ���
            T = zeros(1,K); A = zeros(1,K); B = zeros(1,K); Z = zeros(1,K);
            
            %% ��ÿһ��ά�ȣ��������ŵ�stump����
            for k = 1:K
                [T(k),A(k),B(k),Z(k)] = obj.select_stump(points(k,:),labels,weight);
            end
            
            %% �趨stump�Ĳ���
            wc = learn.Stump();
            [~, wc.k] = max(Z);
            wc.t = T(wc.k); wc.a = A(wc.k); wc.b = B(wc.k);
        end
    end
    
    methods(Access = public)
        %% �о�
        function [y,F] = predict(obj,points)
            %PREDICT �����ݵ���з��࣬����Ϊ1������Ϊ0
            % ʹ�þ���ѵ����ģ���ж����ݵ�ķ���
    
            %% ��ʼ��
            M = length(obj.weak); % M���������ĸ���
            [~,N] = size(points); % N���ݵ���
            f = zeros(M,N); % �洢���������ķ�����
            
            %% ���������������������
            for m=1:M
                f(m,:) = obj.weak{m}.compute(points); 
            end
            
            %% �ۼ�����������������о�
            F = obj.alfa * f;
            y = F > 0;
        end
        
        %% ѵ��
        function obj = train(obj,points,labels,M)
            % train ѵ��GentleAdaBoostģ��
            % ���룺
            % points ���ݵ�
            % labels ��ǩ��+1��-1
            % M ���������ĸ���
            % �����
            % obj ����ѵ����boost����
            
            %% ��ʼ��
            [~,N] = size(points); % �õ����ݵ���Ŀ
            Fx = zeros(1,N); % ������¼����Fx��ֵ
            weight = ones(1,N) / N; % ��ʼ��Ȩֵ
            
            %% ����
            for m = 1:M
                %% ѡ�����ŵ���������
                % ����������fm��һ��stump��������4������ȷ����(a,b,k,t)
                % fm = a * (x(k) > t) + b
                wc = obj.select_wc(points,labels,weight);
                
                %% �����������������뵽ǿ��������
                obj.weak{1+length(obj.weak)} = wc;
                
                %% ������������������������Ȩֵ������ǿ�������ĺ���ֵ
                fm = wc.compute(points); % �����������������
                err = weight * xor((fm > 0),(labels > 0))';
                err = max(err,1e-100);
                c = 0.5 * log((1 - err)/err);
                obj.alfa = [obj.alfa c];
                Fx = Fx + c.*fm; % ����ǿ������
                
                %% ���²���һ��Ȩֵ
                weight = weight .* exp(-c.*labels.*fm); % ����Ȩֵ
                weight = weight ./ sum(weight); % ��һ��Ȩֵ
                
                %% ���������
                disp(sum(xor(Fx>0,labels>0)) / N);
                
                %% ��ͼ
                if wc.k == 1
                    x0 = 1; y0 = 0; z0 = -wc.t;
                else
                    x0 = 0; y0 = 1; z0 = -wc.t;
                end
                f = @(x,y) x0*x+y0*y+z0;
                ezplot(f,[min(points(1,:)),max(points(1,:)),min(points(2,:)),max(points(2,:))]);
                drawnow;
            end
        end
    end
    
    %% ��Ԫ����
    methods(Static)
        function boost = unit_test()
            clear all;
            close all;
            rng(2)
            
            boost = learn.DiscreteAdaBoost();
            
            N = 1e3;
            [points,labels] = learn.GenerateData.type4(N);
            
            figure;
            group1 = points(:,labels== 1);
            group2 = points(:,labels==-1);
            plot(group1(1,:),group1(2,:),'+'); hold on;
            plot(group2(1,:),group2(2,:),'.'); 
            
            M = 300;
            boost = boost.train(points,labels,M);
        end
    end
end


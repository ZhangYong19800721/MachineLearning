classdef GentleAdaBoostSSCPro
    %GentleAdaBoostSSCPro Gentle AdaBoost Pro Similarity Sensitive Coding�㷨ʵ��
    %  �ο�����"Learning Task-specific Similarity" 2005, Chapter 3, by Gregory
    %  Shakhnarovich
    
    properties
        weak; % �������ɸ�����������cell����
    end
    
    methods(Access = private)
         %% ѡ��������������
        function [t,a,b,err] = select_stump(obj,points,labels,weight)
            %select_stump �ڸ���ά�ȵ������ѡ�����ŵ�stump����
            % ��������fm��һ��stump��������4������ȷ����(a,b,k,t)
            % fm = a * [(x1(k) > t) == (x2(k) > t)] + b
            % ���룺
            %   points ���ݵ㣨ֻ������kά�����ݣ�ֻ��һ�У�
            %   labels ��ǩ,����3��(��1/2�������ݵ��±꣬��3�������Ʊ�ǩ+1��-1,ÿһ�б�ʾһ��������)
            %   weight Ȩֵ,��Ϊ1�ķǸ���������labels��������ͬ
            % �����
            %   t ����
            %   a a+b��ʾ������������Ϊ�������������ݵ�ģ���Ȩ������� - ��Ȩ�Ǽ�����ʣ�
            %   b   b��ʾ������������Ϊ�������������ݵ�ģ���Ȩ������� - ��Ȩ�Ǽ�����ʣ�
            %   err ���ֵ err = sum(w * |z - (a*[(x1(k)>t)==(x2(k)>t)] + b)|^2)
            
            %% ��ʼ��
            I = labels(1,:);      % ��������1
            J = labels(2,:);      % ��������2
            L = labels(3,:);      % ��ǩ
            
            %% �������п��ܵ�����ֵT
            T = unique(points); % ȥ���ظ�ֵ������
            T = [T(1)-eps, (T(1:(end-1)) + T(2:end))/2,T(end)+eps]; % �õ����п��ܵ�����ֵ
            
            %% �����п��ܵ�����ֵ����a��b��ֵ
            A = zeros(size(T)); B = zeros(size(T));
            for m = 1:length(T)
                t = T(m); pos = points > t; i = pos(I)==pos(J);
                B(m) = sum(L(~i) .* weight(~i)) / max(1e-100,sum(weight(~i)));
                A(m) = sum(L( i) .* weight( i)) / max(1e-100,sum(weight( i))) - B(m);
            end
            
            %% �������
            % ���ļ��㷽ʽΪ
            %   error = sum(w.*(z-(a(i)*[(x1>t(i))==(x2>t(i))]+b(i))).^2);
            E = zeros(size(T)); x1 = points(I); x2 = points(J); 
            for m = 1:length(T)
                E(m) = sum(weight .* (L - (A(m)*((x1>T(m))==(x2>T(m)))+B(m))).^2);
            end
            
            %% ��������������ŵ�����
            [err, best.i] = min(E);
            t = T(best.i);
            a = A(best.i);
            b = B(best.i);
        end
        
        %% ѡ��������������
        function wc = select_wc(obj,points,labels,weight)
            %select_wc ѡ�����ŵ���������
            % ��������fm��һ�����κ�����f(x)=x'*A*x+B*x+C
            % fm = a * [(f(x1) > t) == (f(x2) > t)] + b
            % ���룺
            %   points ���ݵ�
            %   labels ��ǩ,����3��(��1/2�������ݵ��±꣬��3�������Ʊ�ǩ+1��-1,ÿһ�б�ʾһ��������)
            %   weight Ȩֵ,��Ϊ1�ķǸ�����
            % �����
            %   wc ���ŵ���������
            
            %% ��ʼ��
            [K,N] = size(points); % K���ݵ�ά�ȣ�N���ݵ���
            [~,Q] = size(labels); % ���ݶԵ���Ŀ
            T = zeros(1,K); A = zeros(1,K); B = zeros(1,K); E = zeros(1,K);
            
            %% ��ÿһ��ά�ȣ��������ŵ�stump����
            for k = 1:K
                [T(k),A(k),B(k),E(k)] = obj.select_stump(points(k,:),labels,weight);
            end
            [~, best.k] = min(E); best.t = T(best.k); best.a = A(best.k); best.b = B(best.k);
            
            %% ����Ѱ�Ŷ��κ����Ĳ���
            wc = learn.QuadraticSSC(); %��ʼ�����κ����Ĳ���ΪStump�õ��Ľ��������Ѱ�ŵĽ�������Stump�Ľ������
            wc.A = zeros(K); wc.B = zeros(1,K); wc.B(best.k) = 1; wc.C = -best.t; wc.a = best.a; wc.b = best.b; 
        end
    end
    
    methods(Access = public)
        %% ��ϣ����
        function c = hash(obj,points)
            %hash �������ݵ�Ĺ�ϣ��
            % ��hash���Ǳ����������Ե�hash�� 
            
            %% ��ʼ��
            M = length(obj.weak); % M���������ĸ���
            [~,N] = size(points); % N���ݵ���
            c = false(M,N);       % ���hash��ı�����ÿһ�ж�Ӧһ�����ݵ�
            
            %% �������ݵ��Ӧ��M������
            for m = 1:M
                f = obj.weak{m}.f; % ��m�����������ı任����
                v = f.compute(points); % �����ݵ�任��ĺ���ֵ
                c(m,:) = v > 0; % ������Ϊ1����������Ϊ0
            end
        end
        
        %% �о�
        function [y,F] = predict(obj,points,pairs)
            %PREDICT �ж��������ݵ��Ƿ����ƣ�����Ϊ��1��������Ϊ��0
            % 
            
            %% ��ʼ��
            M = length(obj.weak); % M���������ĸ���
            [~,Q] = size(pairs);  % Q���ݶ���
            f = zeros(M,Q); % �洢���������ķ�����
            
            %% ���������������������
            for m=1:M
                f(m,:) = obj.weak{m}.compute(points,pairs);
            end
            
            %% �ۼ�����������������о�
            F = sum(f,1);
            y = F > 0;
        end
        
        %% ѵ��
        function obj = train(obj,points,labels,M)
            % train ѵ��BoostSSCģ��
            % ���룺
            % points ���ݵ�
            % labels ��ǩ������3��(��1/2�������ݵ��±꣬��3�������Ʊ�ǩ+1��-1,ÿ1�б�ʾһ��������)
            % M ���������ĸ���
            % �����
            % obj ����ѵ����BoostSCC����
            
            %% ��ʼ��
            [~,Q] = size(labels); % �õ����ݶ���Ŀ
            I = labels(1,:); J = labels(2,:); L = labels(3,:); P = [I;J];
            Fx = zeros(1,Q); % ������¼����Fx��ֵ
            weight = ones(1,Q) / Q; % ��ʼ��Ȩֵ
            
            %% ����
            for m = 1:M
                %% ѡ�����ŵ���������
                % ��������fm��һ�����κ���������f(x) = x'*A*x+B*x+C
                % fm = a * [(f(x1) > t) == (f(x2) > t)] + b
                wc = obj.select_wc(points,labels,weight);
                
                %% �����������������뵽ǿ��������
                obj.weak{1+length(obj.weak)} = wc;
                
                %% �������������������������ǿ�������ĺ���ֵ
                fm = wc.compute(points,P); % �����������������
                Fx = Fx + fm; % ����ǿ������
                
                %% ���²���һ��Ȩֵ
                weight = weight .* exp(-L.*fm); % ����Ȩֵ
                weight = weight ./ sum(weight); % ��һ��Ȩֵ
                
                %% ���������
                y = obj.predict(points,P); l = L>0;
                error = sum(y~=l) / Q;
                disp(num2str(error));
                
                %% ��ͼ
                A = wc.A; B = wc.B; C = wc.C;
                f = @(x,y) [x y]*A*[x y]' + B*[x y]' + C;
                ezplot(f,[min(points(1,:)),max(points(1,:)),min(points(2,:)),max(points(2,:))]);
                drawnow;
            end
        end
    end
    
    %% ��Ԫ����
    methods(Static)
        function ssc = unit_test()
            clear all;
            close all;
            rng(2)
            
            ssc = learn.GentleAdaBoostSSCPro();
            
            N = 400;
            [points,labels] = learn.GenerateData.type8(N);
            
            M = 10;
            ssc = ssc.train(points,labels,M);
        end
    end
end


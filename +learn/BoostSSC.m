classdef BoostSSC
    %BoostSSC Boost Similarity Sensitive Coding�㷨ʵ��
    %  �ο�����"Small Codes and Large Image Databases for Recognition" Section
    %  2.1
    
    properties
        weak; % �������ɸ�����������cell����
    end
    
    methods(Access = private)
         %% ѡ��������������
        function [t,a,b,err] = select_stump(obj,points,labels,weight)
            %select_stump �ڸ���ά�ȵ������ѡ�����ŵ�stump����
            % ����������fm��һ��stump��������4������ȷ����(a,b,k,t)
            % fm = a * [(x1(k) > t) == (x2(k) > t)] + b
            % ���룺
            %   points ���ݵ㣨ֻ������kά�����ݣ�ֻ��һ�У�
            %   labels ��ǩ,����3��(��1/2�������ݵ��±꣬��3�������Ʊ�ǩ+1��-1,ÿһ�б�ʾһ��������)
            %   weight Ȩֵ,��Ϊ1�ķǸ���������labels��������ͬ
            % �����
            %   t ����
            %   a ������ֵ�Ҳ�� ��Ȩ������� - ��Ȩ�Ǽ������
            %   b ������ֵ���� ��Ȩ������� - ��Ȩ�Ǽ������
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
                t = T(m); pos = points > t; i = xor(pos(I), pos(J));
                B(m) = sum(L( i) .* weight( i)) / max(1e-100,sum(weight( i)));
                A(m) = sum(L(~i) .* weight(~i)) / max(1e-100,sum(weight(~i))) - B(m);
            end
            
            %% �������
            % ���ļ��㷽ʽΪ
            %   error = sum(w.*(z-(a(i)*[(x1>t(i))==(x2>t(i))]+b(i))).^2);
            E = zeros(size(T)); x1 = points(I); x2 = points(J); 
            for m = 1:length(T)
                E(m) = weight * ((L - (A(m)*((x1>T(m))==(x2>T(m)))+B(m))).^2)';
            end
            
            %% ��������������ŵ�����
            [err, best.i] = min(E);
            t = T(best.i);
            a = A(best.i);
            b = B(best.i);
        end
        
        %% ѡ��������������
        function wc = select_wc(obj,points,labels,weight)
            %select_wc ѡ�����ŵ�stump��������
            % ��������fm��һ��stump��������4������ȷ����(a,b,k,t)
            % fm = a * [(x1(k) > t) == (x2(k) > t)] + b
            % ���룺
            %   points ���ݵ�
            %   labels ��ǩ,����3��(��1/2�������ݵ��±꣬��3�������Ʊ�ǩ+1��-1,ÿһ�б�ʾһ��������)
            %   weight Ȩֵ,��Ϊ1�ķǸ�����
            % �����
            %   wc ���ŵ���������
            
            %% ��ʼ��
            [K,N] = size(points); % K���ݵ�ά�ȣ�N���ݵ���
            [~,Q] = size(labels); % ���ݶԵ���Ŀ
            t = zeros(1,K); a = zeros(1,K); b = zeros(1,K); err = zeros(1,K);
            
            %% ��ÿһ��ά�ȣ��������ŵ�stump����
            for k = 1:K
                [t(k),a(k),b(k),err(k)] = obj.select_stump(points(k,:),labels,weight);
            end
            
            %% �趨stump�Ĳ���
            wc = learn.StumpSSC();
            [~, wc.k] = min(err);
            wc.t = t(wc.k); wc.a = a(wc.k); wc.b = b(wc.k);
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
                k = obj.weak{m}.k; t = obj.weak{m}.t;
                c(m,:) = points(k,:) > t;
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
            [~,N] = size(points); % �õ����ݵ���Ŀ
            [~,Q] = size(labels); % �õ����ݶ���Ŀ
            I = labels(1,:);
            J = labels(2,:);
            L = labels(3,:);
            P = [I;J];
            Fx = zeros(1,Q); % ������¼����Fx��ֵ
            weight = ones(1,Q) / Q; % ��ʼ��Ȩֵ
            
            %% ����
            for m = 1:M
                %% ѡ�����ŵ���������
                % ����������fm��һ��stump��������4������ȷ����(a,b,k,t)
                % fm = a * [(x1(k) > t) == (x2(k) > t)] + b
                wc = obj.select_wc(points,labels,weight);
                wc.t = 0; wc.k = 2; wc.a = 2; wc.b = -1;
                
                %% �����������������뵽ǿ��������
                obj.weak{1+length(obj.weak)} = wc;
                
                %% �������������������������ǿ�������ĺ���ֵ
                fm = wc.compute(points,P); % �����������������
                Fx = Fx + fm; % ����ǿ������
                
                %% ���²���һ��Ȩֵ
                weight = weight .* exp(-L.*fm); % ����Ȩֵ
                weight = weight ./ sum(weight); % ��һ��Ȩֵ
                
                %% ���������
                y = obj.predict(points,P);
                disp(strcat('error_rate: ',num2str(sum(xor(y,L>0)) / Q)));
            end
        end
    end
    
    %% ��Ԫ����
    methods(Static)
        function ssc = unit_test()
            clear all;
            close all;
            rng(2)
            
            ssc = learn.BoostSSC();
            
            N = 40;
            [points,labels] = learn.GenerateData.type5(N);
            
            M = 64;
            ssc = ssc.train(points,labels,M);
        end
    end
end


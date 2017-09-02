classdef GentleAdaBoost
    %GENTLEADABOOST Gentle AdaBoost�㷨ʵ��
    %  �ο�����"Additive Logistic Regression a Statistical View of Boosting"
    %  ��ƪ���ķǳ���Ҫ����ʾ��boost�㷨�ͼ���ͳ��ģ���ڱ���������ͬ�ģ�����
    %  ��AdaBoost��RealAdaBoost��LogitBoost��GentleAdaBoost�㷨��
    
    properties
        weak; % �������ɸ�����������cell����
    end
    
    methods(Access = private)
         %% ѡ��������������
        function [t,a,b,err] = select_stump(obj,points,labels,weight)
            %select_stump �ڸ���ά�ȵ������ѡ�����ŵ�stump����
            % ��������fm��һ��stump��������4������ȷ����(a,b,k,t)
            % fm = a * (x(k) > t) + b
            % ���룺
            %   points ���ݵ㣨ֻ������kά�����ݣ�ֻ��һ�У�
            %   labels ��ǩ,+1��-1
            %   weight Ȩֵ,��Ϊ1�ķǸ�����
            % �����
            %   t ����
            %   a a+b����������ֵ�Ҳ��(��Ȩ�������-��Ȩ�Ǽ������)
            %   b b  ����������ֵ����(��Ȩ�������-��Ȩ�Ǽ������)
            %   err ���ֵ err = sum(w * |z - (a*(x>th) + b)|^2)
            
            %% ��ʼ��
            [K,N] = size(points); % K���ݵ�ά�ȣ�N���ݵ���
            
            %% �������п��ܵ�����ֵT 
            % ����ǩ��Ȩֵ�����ֵ��������
            [T, sort_idx] = sort(points); 
            T = [T(1)-eps,T+eps]; % �õ����п��ܵ�����ֵ
            
            %% �����п��ܵ�����ֵ����a��b��ֵ
            l = labels(sort_idx); w = weight(sort_idx); % ��ǩ��Ȩֵ��������
            Szw = [0 cumsum(l.*w)]; Ezw = Szw(end); Sw  = [1e-100 cumsum(w)]; % �����ۼƺ�
            B = Szw ./ Sw; % ���п��ܵ�bֵ
            A = (Ezw - Szw) ./ max((1-Sw),1e-100) - B; % ���п��ܵ�Aֵ
            
            %% �������
            % ���ļ��㷽ʽΪ
            %   error = sum(w.*(z-(a(i)*(x>th(i))+b(i))).^2);
            % ʵ�ʼ�����ʹ������Ч�ʸ��ߵļ��㷽ʽ
            err = sum(w.*l.^2) - 2*A.*(Ezw-Szw) - 2*B*Ezw + (A.^2 +2*A.*B) .* (1-Sw) + B.^2;
            
            % ��������������ŵ�����
            [err, best.i] = min(err);
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
            t = zeros(1,K); a = zeros(1,K); b = zeros(1,K); err = zeros(1,K);
            
            %% ��ÿһ��ά�ȣ��������ŵ�stump����
            for k = 1:K
                [t(k),a(k),b(k),err(k)] = obj.select_stump(points(k,:),labels,weight);
            end
            
            %% �趨stump�Ĳ���
            wc = learn.Stump();
            [~, wc.k] = min(err);
            wc.t = t(wc.k); wc.a = a(wc.k); wc.b = b(wc.k);
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
            F = sum(f,1);
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
                
                %% �������������������������ǿ�������ĺ���ֵ
                fm = wc.compute(points); % �����������������
                Fx = Fx + fm; % ����ǿ������
                
                %% ���²���һ��Ȩֵ
                weight = weight .* exp(-labels.*fm); % ����Ȩֵ
                weight = weight ./ sum(weight); % ��һ��Ȩֵ
                
                %% ���������
                y = obj.predict(points);
                disp(sum(xor(y,labels>0)) / N);
            end
        end
    end
    
    %% ��Ԫ����
    methods(Static)
        function boost = unit_test()
            clear all;
            close all;
            rng(2)
            
            boost = learn.GentleAdaBoost();
            
            N = 1e4;
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


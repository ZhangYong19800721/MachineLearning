classdef DiscreteAdaBoostSSCPro
    %DiscreteAdaBoostSSCPro DiscreteAdaBoostSSCPro�㷨ʵ��
    %  �ο�����"Learning Task-specific Similarity" 2005, Chapter 3, by Gregory
    %  Shakhnarovich
    
    properties
        weak; % �������ɸ�����������cell����
        alfa; % ÿ������������ͶƱȨֵ
    end
    
    methods(Access = private)
         %% ѡ��������������
        function [t,z] = select_stump(obj,points,labels,weight)
            %select_stump �ڸ���ά�ȵ������ѡ�����ŵ�stump����
            % ��������fm��һ��stump��������4������ȷ����(a,b,k,t)
            % fm = a * [(x1(k) > t) == (x2(k) > t)] + b
            % ����DiscreteAdaBoost(a=+2��b=-1)����stump�������ص�ֵֻ����+1��-1
            % ���룺
            %   points ���ݵ㣨ֻ������kά�����ݣ�ֻ��һ�У�
            %   labels ��ǩ,����3��(��1/2�������ݵ��±꣬��3�������Ʊ�ǩ+1��-1,ÿһ�б�ʾһ��������)
            %   weight Ȩֵ,��Ϊ1�ķǸ���������labels��������ͬ
            % �����
            %   t ����
            %   z ���������Ż���Ŀ�꺯��z=sum(weight.*labels.*fm)
            
            %% ��ʼ��
            [K,N] = size(points); % K���ݵ�ά�ȣ�N���ݵ���
            I = labels(1,:); J = labels(2,:); L = labels(3,:); P = [I;J];
            
            %% �������п��ܵ�����ֵT 
            T = unique(points); % ȥ���ظ�ֵ������
            T = [T(1)-eps, (T(1:(end-1)) + T(2:end))/2, T(end)+eps]; % �õ����п��ܵ�����ֵ
            
            %% �����п��ܵ�����ֵ����a,b,z
            Z = zeros(size(T));
            for m = 1:length(T)
                t = T(m); pos = points > t;
                p =  2 * (pos(I) == pos(J)) - 1; 
                Z(m) = sum(weight.*L.*p);
            end
            
            %% ��������������ŵ�����
            [~, best.i] = max(abs(Z));
            z = Z(best.i); t = T(best.i);
        end
        
        %% ѡ��������������
        function wc = select_wc(obj,points,labels,weight)
            %select_wc ѡ�����ŵ���������
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
            
            %% ����Ѱ�Ŷ��κ����Ĳ���
            wc = learn.ssc.QuadraticSSC();
            A = [2 0; 0 2]; B = [0 0]; C = -25;
            % A = randn(K); B = randn(1,K); C = randn(1);
            % f = 0.5 * sum((points' * A) .* points',2)' + B * points + repmat(C,1,N);
            % C = -median(f);
            x0 = [reshape(A,[],1); reshape(B,[],1); C];
            F = learn.ssc.DAB_SSC_Pro_Aid(weight,points,labels);

            parameters.learn_rate = 1e-2; % ѧϰ�ٶ�
            parameters.momentum = 0; % ���ٶ���
            parameters.epsilon = 1e-3; % ���ݶȵķ���С��epsilonʱ��������
            parameters.max_it = 5e3; % ����������
            x = learn.optimal.maximize_g(F,x0,parameters);
            
%             parameters.epsilon = 1e-3; %���ݶ�ģС��epsilonʱֹͣ����
%             parameters.alfa = 1e+3; %�����������䱶��
%             parameters.beda = 1e-8; %����������ֹͣ����
%             parameters.max_it = 2e3; %����������
%             parameters.reset = 200; %��������
%             x = learn.optimal.maximize_cg(F,x0,parameters);
            wc.A = reshape(x(1:(K*K)),K,K); wc.B = reshape(x(K*K+(1:K)),1,[]); wc.C = x(end); wc.a = 2; wc.b = -1; 
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
                A = obj.weak{m}.A; B = obj.weak{m}.B; C = obj.weak{m}.C;
                c(m,:) = (0.5 * sum((points' * A) .* points',2)' + B * points + repmat(C,1,N)) > 0;
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
            F = obj.alfa * f;
            y = F > 0;
        end
        
        %% ѵ��
        function obj = train(obj,points,labels,M)
            % train ѵ��DiscreteAdaBoostSSCProģ��
            % ���룺
            % points ���ݵ�
            % labels ��ǩ������3��(��1/2�������ݵ��±꣬��3�������Ʊ�ǩ+1��-1,ÿ1�б�ʾһ��������)
            % M ���������ĸ���
            % �����
            % obj ����ѵ����DiscreteAdaBoostSSCPro����
            
            %% ��ʼ��
            [~,N] = size(points); % �õ����ݵ���Ŀ
            [~,Q] = size(labels); % �õ����ݶ���Ŀ
            I = labels(1,:); J = labels(2,:); L = labels(3,:); P = [I;J];
            Fx = zeros(1,Q); % ������¼����Fx��ֵ
            weight = ones(1,Q) / Q; % ��ʼ��Ȩֵ
            
            %% ����
            for m = 1:M
                %% ѡ�����ŵ���������
                % ��������fm��һ��stump��������4������ȷ����(a,b,k,t)
                % fm = a * [(f(x1) > t)==(f(x2) > t)] + b
                wc = obj.select_wc(points,labels,weight);
                
                %% �����������������뵽ǿ��������
                obj.weak{1+length(obj.weak)} = wc;
                
                %% ������������������������Ȩֵ������ǿ�������ĺ���ֵ
                fm = wc.compute(points,P); % �����������������
                err = weight * xor((fm > 0),(L > 0))';
                err = max(err,1e-100);
                c = 0.5 * log((1 - err)/err);
                obj.alfa = [obj.alfa c];
                Fx = Fx + c.*fm; % ����ǿ������
                
                %% ���²���һ��Ȩֵ
                weight = weight .* exp(-c.*L.*fm); % ����Ȩֵ
                weight = weight ./ sum(weight); % ��һ��Ȩֵ
                
                %% ���������
                disp(sum(xor(Fx>0,L>0)) / Q);
                
                %% ��ͼ
                A = wc.A; B = wc.B; C = wc.C;
                f = @(x,y) 0.5*[x y]*A*[x y]' + B*[x y]' + C;
                warning('off','MATLAB:ezplotfeval:NotVectorized');
                % ezplot(f,[min(points(1,:)),max(points(1,:)),min(points(2,:)),max(points(2,:))]);
                ezplot(f,[-10,10,-10,10]);
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
            
            ssc = learn.ssc.DiscreteAdaBoostSSCPro();
            
            N = 400;
            [points,labels] = learn.data.GenerateData.type9(N);
            
            M = 3;
            ssc = ssc.train(points,labels,M);
        end
    end
end


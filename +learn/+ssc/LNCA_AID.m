classdef LNCA_AID
    %LNCA_AID �����ڷ������������Ż�������
    %  ���㺯��ֵ���ݶ�
    
    properties(Access=private)
        points; % ���ݵ�
        labels; % ���Ʊ�ǩ
        simset; % ���Ƽ���
        lnca;   % ������
    end
    
    methods
        %% ���캯��
        function obj = LNCA_AID(points,labels,lnca) 
            obj.points = points;
            obj.labels = labels;
            obj.lnca = lnca;
            
            %% �������Ƽ���
            [~,N] = size(points);
            I = labels(1,:); J = labels(2,:);
            parfor n = 1:N
                index = (J == n | I == n);
                simset{n} = setdiff(union(I(index),J(index)),n);
            end
            obj.simset = simset;
        end
    end
    
    methods
        %% ����Ŀ�꺯��
        function y = object(obj,x)
            %% ��ʼ��
            [~,N] = size(obj.points);
            obj.rnlnca.weight = x; % �������ò���
            
            %% ����ʵ������
            code = obj.lnca.encode(obj.points); % ����ʵ������
            
            %% ����Ŀ�꺯��
            y = 0;
            for a = 1:N
                D = sum((code - repmat(code(:,a),1,N)).^2,1); % ����a�㵽����������ľ���
                E = exp(-D); E(a) = 0; S = sum(E); % ���㸺ָ������
                b = obj.simset{a};
                y = y + sum(E(b) / S);
            end
        end
        
        %% �����ݶ�
        function g = gradient(obj,x)
            %% ��ʼ��
            [D,N] = size(obj.points); % N��������
            P = size(obj.lnca.weight,1); % P��������
            obj.rnlnca.weight = x; % �������ò���
            g = zeros(P,1); % ��ʼ���ݶ�
            [A,r] = obj.lnca.getw(1); % ���Ա任����
            gA = zeros(size(A(:))); 
            
            %% ����ʵ������
            code = obj.rnlnca.encode(obj.points); 
            
            %% �����ݶ�
            xpoints = obj.points; 
            xsimset = obj.simset;
            parfor a = 1:N
                d_az = sum((code - repmat(code(:,a),1,N)).^2,1); % ����a�㵽����������ľ���
                e_az = exp(-d_az); e_az(a) = 0; s_az = sum(e_az); % ���㸺ָ������
                p_az = e_az / s_az; % ����a�㵽����������ĸ���
                x_az = repmat(sqrt(p_az),D,1) .* (repmat(xpoints(:,a),1,N) - xpoints); % Xaz
                gA = gA + sum(p_az(xsimset{a})) * (x_az*x_az') - x_az(:,xsimset{a})*x_az(:,xsimset{a})';
            end
            
            gA = 2*A*gA;
            g(r) = gA(:);
        end
    end
    
    methods(Static)
        %% ��Ԫ����
        function [] = unit_test() 
            
        end
    end
end


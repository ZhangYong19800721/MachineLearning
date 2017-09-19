classdef LNCA_AID
    %LNCA_AID �����ڷ������������Ż�������
    %  ���㺯��ֵ���ݶ�
    
    properties(Access=private)
        points; % ���ݵ�
        labels; % ���Ʊ�ǩ
        simset; % ���Ƽ���
        difset; % ��ͬ����
        lnca;   % ������
    end
    
    methods
        %% ���캯��
        function obj = LNCA_AID(points,labels,lnca) 
            obj.points = points;
            obj.labels = labels;
            obj.lnca = lnca;
            
            labels_pos = labels(:,labels(3,:)==+1);
            labels_neg = labels(:,labels(3,:)==-1);
            
            %% �������Ƽ���
            [~,N] = size(points);
            I = labels_pos(1,:); J = labels_pos(2,:);
            parfor n = 1:N
                index = (J == n | I == n);
                simset{n} = setdiff(union(I(index),J(index)),n);
            end
            obj.simset = simset;
            
            %% ���㲻ͬ����
            I = labels_neg(1,:); J = labels_neg(2,:);
            parfor n = 1:N
                index = (J == n | I == n);
                difset{n} = setdiff(union(I(index),J(index)),n);
            end
            obj.difset = difset;
        end
    end
    
    methods
        %% ����Ŀ�꺯��
        function y = object(obj,x)
            %% ��ʼ��
            [~,N] = size(obj.points);
            obj.lnca.weight = x; % �������ò���
            
            %% ����ʵ������
            code = obj.lnca.encode(obj.points); % ����ʵ������
            
            %% ����Ŀ�꺯��
            y = 0; 
            for a = 1:N
                d1 = sum((code(:,obj.simset{a}) - repmat(code(:,a),1,numel(obj.simset{a}))).^2,1);
                d2 = sum((code(:,obj.difset{a}) - repmat(code(:,a),1,numel(obj.difset{a}))).^2,1);
                e1 = exp(-d1); se1 = sum(e1);
                e2 = exp(-d2); se2 = sum(e2);
                y = y + se1 / (se1 + se2);
            end
            y = y / N;
        end
        
        %% �����ݶ�
        function g = gradient(obj,x)
            %% ��ʼ��
            [D,N] = size(obj.points); % N��������
            P = size(obj.lnca.weight,1); % P��������
            obj.lnca.weight = x; % �������ò���
            g = zeros(P,1); % ��ʼ���ݶ�
            [A,r] = obj.lnca.getw(1); % ���Ա任����
            
            %% ����ʵ������
            code = obj.lnca.encode(obj.points); 
            
            %% �����ݶ�
            q = 0;
            for a = 1:N
                d1 = sum((code(:,obj.simset{a}) - repmat(code(:,a),1,numel(obj.simset{a}))).^2,1); % ����a�㵽���Ƶ�ľ���
                d2 = sum((code(:,obj.difset{a}) - repmat(code(:,a),1,numel(obj.difset{a}))).^2,1);
                e1 = exp(-d1); se1 = sum(e1); % ���㸺ָ������
                e2 = exp(-d2); se2 = sum(e2);
                p1 = e1 / (se1 + se2); % ����a�㵽���Ƶ�ĸ���
                p2 = e2 / (se1 + se2); 
                x_a1 = repmat(obj.points(:,a),1,numel(obj.simset{a})) - obj.points(:,obj.simset{a});
                x_a2 = repmat(obj.points(:,a),1,numel(obj.difset{a})) - obj.points(:,obj.difset{a});
                q1 = x_a1 * diag(p1) * x_a1';
                q2 = x_a2 * diag(p2) * x_a2';
                q = q + (se1 / (se1 + se2)) * (q1 + q2) - q1;
            end
            
            q = 2 * A * q / N;
            g(r) = q(:);
        end
    end
    
    methods(Static)
        %% ��Ԫ����
        function [] = unit_test() 
            
        end
    end
end


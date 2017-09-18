classdef NCA_AID
    %NCA_AID ����������ڷ������������Ż�������
    %  ����������ĺ���ֵ���ݶ�
    
    properties(Access=private)
        points; % ���ݵ�
        labels; % ���Ʊ�ǩ
        simset; % ���Ƽ���
        rnlnca; % ������
    end
    
    methods
        %% ���캯��
        function obj = R_NL_NCA_AID(points,labels,rnlnca) 
            obj.points = points;
            obj.labels = labels;
            obj.rnlnca = rnlnca;
            
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
            
            %% �������ò���
            obj.rnlnca.weight = x;
            
            %% ����ʵ������
            code = obj.rnlnca.encode(obj.points,'real');
            y = 0;
            for a = 1:N
                dist = sum((code - repmat(code(:,a),1,N)).^2); % a�㵽�������е�֮��ľ���
                prob = exp(-dist);
                for b = obj.simset{a}
                    p = prob(b) / (sum(prob) - 1);
                    y = y + p;
                end
            end
        end
        
        %% �����ݶ�
        function g = gradient(obj,x)
            %% ��ʼ��
            [~,N] = size(obj.points);
            
            %% �������ò���
            obj.rnlnca.weight = x;
            
            
        end
    end
    
    methods(Static)
        %% ��Ԫ����
        function [] = unit_test() 
            disp('unit-test of learn.ssc.R_NL_NCA_AID');
        end
    end
end


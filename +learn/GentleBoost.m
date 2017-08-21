classdef GentleBoost
    %GENTLEBOOST GentleBoost�㷨
    %  �ο���Robust Real-time Object Detection��
    
    properties
    end
    
    methods
        function [obj,w]  = ensemble(obj,points,labels,weight,wc)
            %ensemble �����������
            %   ���룺
            %   points ��ʾ���ݵ�
            %   labels ��ǩ1��0
            %   weight ���ʷֲ�Ȩֵ
            %   wc �¼������������
            % 
            %   �����
            %   obj ѵ�����adaboost����
            %   w ���º��Ȩֵ
            
            c = wc.predict(points); % ʹ���������������е����ݵ���з���
            epsilon = sum(weight .* abs(c - labels)); % ������weight�ֲ������µķ������
            beda = epsilon / (1 - epsilon); 
            obj.weak{1+length(obj.weak)} = wc;
            obj.alfa = [obj.alfa log(1/beda)];
            w = weight .* (beda.^(1 - c~=labels)); % ����Ȩֵ
            w = w ./ sum(w);
        end
        
        function y = predict(obj,points)  % ʹ�þ���ѵ����ģ���ж����ݵ�ķ���
            %PREDICT �����ݵ���з��࣬����Ϊ1������Ϊ0
            %
            H = length(obj.alfa); [~,N] = size(points); % H���������ĸ�����N���ݵ���
            c = logical(H,N); % �洢���������ķ�����
            for h=1:H, c(h,:) = obj.weak{h}.predict(points); end
            y = obj.alfa * c >= sum(obj.alfa) / 2;
        end
    end
end


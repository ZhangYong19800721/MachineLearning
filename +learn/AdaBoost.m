classdef AdaBoost
    %AdaBoost ʵ��AdaBoost�㷨
    %  �ο�����"Improved Boosting Algorithms Using Confidence-rated Predictions"
    
    properties
        weak; % �������ɸ�����������cell����
        alfa; % ÿ������������ͶƱȨֵ
    end
    
    methods
        function obj = AdaBoost()
        end
    end
    
    methods
        function [obj,w]  = ensemble(obj,points,labels,weight,wc)
            %ensemble �����������
            %   ���룺
            %   points ��ʾ���ݵ�
            %   labels ��ǩ+1��-1
            %   weight ���ʷֲ�Ȩֵ
            %   wc �¼������������
            % 
            %   �����
            %   obj ѵ�����adaboost����
            %   w ���º��Ȩֵ
            
            c = wc.predict(points);
            r = weight * (labels .* c)';
            beda = 0.5 * log((1 + r)/(1 - r));
            obj.weak{1+length(obj.weak)} = wc;
            obj.alfa = [obj.alfa beda];
            w = weight .* exp(-beda * labels .* c);
            w = w ./ sum(w);
        end
        
        function y = predict(obj,points)  % ʹ�þ���ѵ����ģ���ж����ݵ�ķ���
            %PREDICT �ж��������Ƿ����ƣ�����Ϊ+1��������Ϊ-1
            %
            H = length(obj.alfa); [~,N] = size(points); % H���������ĸ�����N���ݵ���
            c = zeros(H,N); % �洢���������ķ�����
            for h=1:H
                c(h,:) = obj.weak{h}.predict(points);
            end
            y = sign(obj.alfa * c);
        end
    end
end


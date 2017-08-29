classdef GentleBoost
    %GENTLEBOOST GentleBoost�㷨
    %  �ο�����"Additive Logistic Regression a Statistical View of Boosting"
    %  ��ƪ���ķǳ���Ҫ����ʾ��boost�㷨�ͼ���ͳ��ģ���ڱ���������ͬ�ģ�����
    %  ��AdaBoost��RealBoost��LogitBoost��GentleBoost�㷨��
    
    properties
        weak; % �������ɸ�����������cell����
    end
    
    methods(Access = private)
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
            
            f = wc.predict(points); % ���ȼ����������������
            obj.weak{1+length(obj.weak)} = wc; % �������������뵽boost����
            w = weight .* (-labels .* f); % ����Ȩֵ
            w = w ./ sum(w); % ��һ��Ȩֵ
        end
    end
    
    methods(Access = public)
        function y = compute(obj,points)
            M = length(obj.weak); [~,N] = size(points); % M���������ĸ�����N���ݵ���
            f = logical(M,N); % �洢���������ķ�����
            for m=1:M, f(m,:) = obj.weak{m}.predict(points); end
            y = sum(f);
        end
        
        function y = predict(obj,points)  % ʹ�þ���ѵ����ģ���ж����ݵ�ķ���
            %PREDICT �����ݵ���з��࣬����Ϊ1������Ϊ0
            %
            y = obj.compute(points) > 0;
        end
        
        function obj = train(obj,points,labels,M)
            % train ѵ��GentleBoostģ��
            % ���룺
            % points ���ݵ�
            % labels ��ǩ��+1��-1
            % M ���������ĸ���
            % �����
            % obj ����ѵ����boost����
            
            [~,N] = size(points); % �õ����ݵ���Ŀ
            weight = ones(1,N) / N; % ��ʼ��Ȩֵ
            
            
        end
    end
end


classdef DAB_SSC_Pro_Aid
    %DAB_SSC_Pro_Aid DiscreteAdaBoostSSCPro�����Ż��ĸ�����
    %   �˴���ʾ��ϸ˵��
    
    properties
        weight;
        points;
        labels;
    end
    
    methods
        function obj = DAB_SSC_Pro_Aid(weight,points,labels)
            obj.weight = weight;
            obj.points = points;
            obj.labels = labels;
        end
        
        function g = gradient(obj,x)
            %% ��ʼ��
            [K,N] = size(obj.points); Q = K*K+K+1; % K���ݵ�ά�ȡ�N���ݵ�����Q���η��̲�������
            A = reshape(x(1:(K*K)),K,K); B = reshape(x(K*K+(1:K)),1,[]); C = x(end);
            I = obj.labels(1,:); J = obj.labels(2,:); L = obj.labels(3,:);
            
            %% �����ݶ�
            f = learn.tools.quadratic(A,B,C,obj.points); % �������е��f����ֵ
            h = learn.tools.sigmoid(f); % �������е��h����ֵ
            g_h_C = ones(1,N); % h������C���ݶ�
            g_h_B = obj.points; % h������B���ݶ�
            g_h_A = 0.5 * obj.points(reshape(repmat(1:K,K,1),1,[]),:) .* obj.points(repmat(1:K,1,K),:); % h������A���ݶ�
            g_h_x = repmat(h.*(1-h),Q,1) .* [g_h_A;g_h_B;g_h_C];
            g_c_x = 4 * (g_h_x(:,I) .* (g_h_x(:,J) - 0.5)) + 4 * (g_h_x(:,J) .* (g_h_x(:,I) - 0.5));
            g = sum(repmat(obj.weight.*L,Q,1) .* g_c_x,2);
        end
        
        function y = object(obj,x)
            %% ��ʼ��
            [K,N] = size(obj.points);
            A = reshape(x(1:(K*K)),K,K); B = reshape(x(K*K+(1:K)),1,[]); C = x(end);
            I = obj.labels(1,:); J = obj.labels(2,:); L = obj.labels(3,:);
            
            %% ����Ŀ�꺯��ֵ
            f = learn.tools.quadratic(A,B,C,obj.points); % �������е��f����ֵ
            h = learn.tools.sigmoid(f); % �������е��h����ֵ
            c = 4 * (h(I) - 0.5) .* (h(J) - 0.5); % �������е��c����ֵ
            y = sum(obj.weight .* L .* c,2); % ����Ŀ�꺯��ֵ
        end
    end
    
end


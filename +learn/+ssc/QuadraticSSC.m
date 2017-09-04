classdef QuadraticSSC
    %QUADRATICSSC ���BoostProSSCʹ�õ���������
    %   ʵ�ֶ��κ����͵���������
    %   ���κ�������ʽΪf(x) = x'*A*x+B*x+C
    %   ��f(x)>0ʱ��Ϊ��������f(x)<0ʱ��Ϊ����
    
    properties
        A; B; C; a; b;
    end
    
    methods(Access = public)
        function y = compare(obj,points1,points2)
            % compare ����������֮���������ֵ
            % ����������Ϊ����ʱ����a+b
            % ����������Ϊ����ʱ����b
            
            [~,N1] = size(points1); [~,N2] = size(points2); 
            v1 = 0.5 * sum((points1' * obj.A) .* points1',2)' + obj.B * points1 + repmat(obj.C,1,N1);
            v2 = 0.5 * sum((points2' * obj.A) .* points2',2)' + obj.B * points2 + repmat(obj.C,1,N2);
            c1 = v1 > 0;
            c2 = v2 > 0;
            y = obj.a * (c1==c2) + obj.b;
        end
        
        function y = compute(obj,points,paridx)
            % compute ����������֮���������ֵ
            % ����������Ϊ����ʱ����a+b
            % ����������Ϊ����ʱ����b
            
            [~,N] = size(points); 
            v = 0.5 * sum((points' * obj.A) .* points',2)' + obj.B * points + repmat(obj.C,1,N);
            c = v > 0; c1 = c(paridx(1,:)); c2 = c(paridx(2,:));
            y = obj.a * (c1==c2) + obj.b;
        end
    end
end


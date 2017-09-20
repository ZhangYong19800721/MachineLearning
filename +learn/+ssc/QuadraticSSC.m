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
            
            v1 = learn.tools.quadratic(obj.A,obj.B,obj.C,points1);
            v2 = learn.tools.quadratic(obj.A,obj.B,obj.C,points2);
            c1 = v1 > 0;
            c2 = v2 > 0;
            y = obj.a * (c1==c2) + obj.b;
        end
        
        function y = compute(obj,points,paridx)
            % compute ����������֮���������ֵ
            % ����������Ϊ����ʱ����a+b
            % ����������Ϊ����ʱ����b
            
            v = learn.tools.quadratic(obj.A,obj.B,obj.C,points);
            c = v > 0; c1 = c(paridx(1,:)); c2 = c(paridx(2,:));
            y = obj.a * (c1==c2) + obj.b;
        end
    end
end


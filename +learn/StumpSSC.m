classdef StumpSSC
    %StumpSSC ���BoostSSCʹ�õ���������
    %   fm = a * [(x1(k) > t) == (x2(k) > t)] + b
    
    properties
        a; b; k; t;
    end
    
    methods(Access = private)
        function y = compare(obj,points1,points2)
            % compare ����������֮���������ֵ
            c1 = points1(obj.k,:) > obj.t;
            c2 = points2(obj.k,:) > obj.t;
            y = obj.a * xor(c1,c2) + obj.b;
        end
        
        function y = compute(obj,points,paridx)
            % compute ����������֮���������ֵ
            c = points(obj.k,:) > obj.t;
            y = obj.a * xor(c(paridx(1,:)),c(paridx(2,:))) + obj.b;
        end
    end
end


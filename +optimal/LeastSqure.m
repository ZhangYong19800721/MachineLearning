classdef LeastSqure
    %LEASTSQURE �˴���ʾ�йش����ժҪ
    %   �˴���ʾ��ϸ˵��
    
    properties
    end
    
    methods
    end
    
    methods(Static)
        function [x1,z1] = LevenbergMarquardt(f,J,x0,epsilon)
            %LevenbergMarquardt һ�������С�����������ֵ�㷨
            %   ��һ�ָĽ��ĸ�˹ţ�ٷ�����������������ڸ�˹ţ�ٷ���������½�����֮��
            
            alfa = 0.01; beda = 10; D = length(x0);
            
            x1 = x0; z1 = [];
            
            while true
                J1 = J(x1);
                f1 = f(x1);
                G1 = J1' * f1;
                if norm(G1) < epsilon
                    break;
                end
                delta = -1 * (J1'*J1 + alfa * eye(D)) \ G1;
                x2 = x1 + delta;
                f2 = f(x2);
                z1 = norm(f1,2).^2;
                z2 = norm(f2,2).^2;
                if z1 > z2
                    alfa = alfa / beda;
                    x1 = x2;
                else
                    alfa = alfa * beda;
                end
            end
        end
    end
    
    methods(Static) % ��Ԫ����
        function y = unit_test()
            clear all
            close all
            
            F = @(x)([x(1) - x(2); x(1) + x(2)]);
            J = @(x)([1 -1; 1 1]); 
            x0 = [10 10]';
            [x,z] = optimal.LeastSqure.LevenbergMarquardt(F,J,x0,1e-6);
        end
    end
end


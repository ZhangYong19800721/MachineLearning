function [x1,z1] = minimize_lm(F,x0,parameters)
%minimize_lm LevenbergMarquardt�㷨��һ�������С�����������ֵ�㷨
%   ��һ�ָĽ��ĸ�˹ţ�ٷ�����������������ڸ�˹ţ�ٷ���������½�����֮��
%   ���룺
%       F ����F.jacobi(x)����Ŀ�꺯����jacobi���󣬵���F.vector(x)����Ŀ�꺯���ĸ�������ֵf = [f1,f2,f3,...,fn]
%       Ŀ�꺯��Ϊsum(f.^2)
%       x0 ��������ʼλ��
%       parameters.epsilon ���ݶȵķ���С��epsilonʱ��������
%       parameters.max_it ����������
%   �����
%       x ���ŵĲ�����
%       y ��С�ĺ���ֵ

    alfa = 0.01; beda = 10; 
    D = length(x0); % ���ݵ�ά��
    x1 = x0;
    
    for it = 1:parameters.max_it
        j1 = F.jacobi(x1);
        f1 = F.vector(x1);
        g1 = j1' * f1;
        ng1 = norm(g1);
        if ng1 < parameters.epsilon
            break;
        end
        delta = -1 * (j1'*j1 + alfa * eye(D)) \ g1;
        x2 = x1 + delta;
        f2 = F.vector(x2);
        z1 = norm(f1,2).^2;
        z2 = norm(f2,2).^2;
        
        disp(sprintf('Ŀ�꺯��:%f ��������:%d �ݶ�ģ:%f',sum(f1.^2),it,ng1));
        
        if z1 > z2
            alfa = alfa / beda;
            x1 = x2;
        else
            alfa = alfa * beda;
        end
    end
end


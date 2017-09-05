function [x,y] = maximize_lm(F,x0,parameters)
%maximize_lm �ݶȷ�
%   ���룺
%       F ����F.jacobi(x)����Ŀ�꺯����jacobi���󣬵���F.vector(x)����Ŀ�꺯���ĸ�������ֵf = [f1,f2,f3,...,fn]
%       Ŀ�꺯��Ϊsum(f.^2)
%       x0 ��������ʼλ��
%       parameters.epsilon ���ݶȵķ���С��epsilonʱ��������
%       parameters.max_it ����������
%   �����
%       x ���ŵĲ�����
%       y ��С�ĺ���ֵ
    
    F = learn.optimal.NEGATIVE(F);
    [x,y] = learn.optimal.minimize_lm(F,x0,parameters);
    y = -y;
end


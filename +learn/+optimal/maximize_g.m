function [x,y] = maximize_g(F,x0,parameters)
%maximize_g �ݶȷ�
%   ���룺
%       F ����F.gradient(x)����Ŀ�꺯�����ݶȣ�����F.object(x)����Ŀ�꺯����ֵ
%       x0 ��������ʼλ��
%       parameters.learn_rate ѧϰ�ٶ�
%       parameters.momentum ���ٶ���
%       parameters.epsilon ���ݶȵķ���С��epsilonʱ��������
%       parameters.max_it ����������
%   �����
%       x ���ŵĲ�����
%       y ��С�ĺ���ֵ
    
    F = learn.optimal.NEGATIVE(F);
    [x,y] = learn.optimal.minimize_g(F,x0,parameters);
    y = -y;
end


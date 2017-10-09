function [x,y] = maximize_sadam(F,x0,parameters)
%maximize_sadam �ݶȷ�
%   ���룺
%       F ����F.gradient(x)����Ŀ�꺯�����ݶȣ�����F.object(x)����Ŀ�꺯����ֵ
%       x0 ��������ʼλ��
%       parameters ����
%   �����
%       x ���ŵĲ�����
%       y ���ŵĺ���ֵ
    
    F = learn.optimal.NEGATIVE(F);
    [x,y] = learn.optimal.minimize_sadam(F,x0,parameters);
    y = -y;
end


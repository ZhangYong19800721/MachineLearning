function [x1,y1] = maximize_cg(F,x0,parameters)
%minimize_cg �����ݶȷ�
% ���룺
%   F �������󣬵���F.object(x)����Ŀ�꺯����x����ֵ������F.gradient(x)����Ŀ�꺯����x�����ݶ�
%   x0 ��������ʼλ��
%   parameters �ο�minimize_cg�Ĳ�������

    F = learn.optimal.NEGATIVE(F);
    [x1,y1] = learn.optimal.minimize_cg(F,x0,parameters);
    y1 = -y1;
end


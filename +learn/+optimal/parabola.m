function [ny,nx] = parabola(F,a,b,parameters)
%PARABOLA ʹ�������߷����о�ȷ������
%  �ο� �������Ż����㷽������MATLAB����ʵ�֡���������ҵ�����磩 
%  ���룺
%       F ������������F.object(x)����Ŀ�꺯����x����ֵ
%       a �����������˵�
%       b ����������Ҷ˵�
%       parameters.parabola.epsilon ֹͣ����������ֵ1e-6
%  �����
%       ny ��С��ĺ���ֵ
%       nx ��С��ı���ֵ

    s0 = x;            s1 = s0 + h;       s2 = s0 + 2*h;
    F0 = F.object(s0); F1 = F.object(s1); F2 = F.object(s2);
    
    h = (3*F0 - 4*F1 + F2) * h / (2*(2*F1 - F0 - F2));
    s0 = s0 + h;
end


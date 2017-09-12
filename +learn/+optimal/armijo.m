function [lamda,nf,nx] = armijo(F,x,g,d,parameters)
%ARMIJO �Ǿ�ȷ��������Armijo׼��
%   �ο� �������Ż����㷽������MATLAB����ʵ�֡���������ҵ�����磩
%   ���룺
%       F ��������F.object(x)����Ŀ�꺯����x����ֵ
%       x ��������ʵλ��
%       g Ŀ�꺯����x�����ݶ�
%       d ��������
%       parameters.beda ֵ������(0,  1)֮��
%       parameters.alfa ֵ������(0,0.5)֮��
%       parameters.maxs �������������������
%
%   �����
%       lamda ��������

    assert(0 < parameters.beda && parameters.beda <   1);
    assert(0 < parameters.alfa && parameters.alfa < 0.5);
    assert(0 < parameters.maxs);
    
    m = 0; f = F.object(x);
    while m <= parameters.maxs
        nx = x + parameters.beda^m * d;
        nf = F.object(nx);
        lamda = parameters.beda^m;
        if nf <= f + parameters.alfa * lamda * g'* d
            break;
        end
        m = m + 1;
    end
end


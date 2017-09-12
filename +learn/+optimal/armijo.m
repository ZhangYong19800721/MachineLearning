function [lamda,nf,nx] = armijo(F,x,g,d,parameters)
%ARMIJO �Ǿ�ȷ��������Armijo׼��
%   �ο� �������Ż����㷽������MATLAB����ʵ�֡���������ҵ�����磩
%   ���룺
%       F ��������F.object(x)����Ŀ�꺯����x����ֵ
%       x ��������ʵλ��
%       g Ŀ�꺯����x�����ݶ�
%       d ��������
%       parameters.armijo.beda ֵ������(0,  1)֮��
%       parameters.armijo.alfa ֵ������(0,0.5)֮��
%       parameters.armijo.maxs �������������������
%
%   �����
%       lamda ��������

    assert(0 < parameters.armijo.beda && parameters.armijo.beda <   1);
    assert(0 < parameters.armijo.alfa && parameters.armijo.alfa < 0.5);
    assert(0 < parameters.armijo.maxs);
    
    m = 0; f = F.object(x);
    while m <= parameters.armijo.maxs
        nx = x + parameters.armijo.beda^m * d;
        nf = F.object(nx);
        lamda = parameters.armijo.beda^m;
        if nf <= f + parameters.armijo.alfa * lamda * g'* d
            break;
        end
        m = m + 1;
    end
end


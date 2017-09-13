function [a,b]=ARR(F,x0,h0,epsilon)
% ARR �Ľ��Ľ��˷�ȷ��������������
% ���룺
%   F �������󣬵���F.object(x)����Ŀ�꺯��
%   x0 ��ʼ����λ��
%   h0 ��ʼ����������h0��������������������
% �����
%   a ����������˵�
%   b ���������Ҷ˵�

    F0 = F.object(x0); 
    x1 = x0 + h0; F1 = F.object(x1);
    
    if F0 > F1
        while F0 > F1
            h0 = 2 * h0; % ���󲽳�
            x1 = x0 + h0; F1 = F.object(x1);
        end
        a = x0;
        b = x1;
    else
        a = x0;
        while F0 <= F1
            h0 = 0.5 * h0; % ��С����
            if abs(h0) < epsilon
                break;
            end
            b  = x1;
            x1 = x0 + h0; F1 = F.object(x1);
        end
    end
end
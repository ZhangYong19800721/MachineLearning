function [a,b]=AR(F,x0,h0)
% AR advance & retreat ���˷�ȷ����������
% ���룺
%   F �������󣬵���F.object(x)����Ŀ�꺯��
%   x0 ��ʼ����λ��
%   h0 ��ʼ��������, h0 > 0
% �����
%   a ����������˵�
%   b ���������Ҷ˵�

    x1 = x0;      F1 = F.object(x1); 
    x2 = x0 + h0; F2 = F.object(x2);
    if F1 > F2
        x = x2; Fx = F2;
        while true
            h0 = 2 * h0; % ���󲽳�
            x1 = x2;
            x2 = x1 + h0; 
            F2 = F.object(x2);
            if F2 > Fx
                break;
            end
        end
    else
        h0 = 0.5 * h0;
        while true
            x2 = 
        end
    end
    
    a=min(x,x2);
    b=max(x,x2);
end
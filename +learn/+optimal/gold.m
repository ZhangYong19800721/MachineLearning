function [ny,nx] = gold(F,a,b,parameters)
%gold ʹ�ûƽ�ָ����������
%  ���룺
%       F ������������F.object(x)����Ŀ�꺯����x����ֵ
%       a �����������˵�
%       b ����������Ҷ˵�
%       parameters.gold.epsilon ֹͣ����������ֵ1e-6
%  �����
%       ny ��С��ĺ���ֵ
%       nx ��С��ı���ֵ

    if nargin <= 3
        parameters.gold.epsilon = 1e-6;
    end
    
    g = (sqrt(5)-1)/2;
    ax = a + (1 - g)*(b - a); Fax = F.object(ax);
    bx = a + g * (b - a);     Fbx = F.object(bx);
    
    while b - a > parameters.gold.epsilon
        if Fax > Fbx
            a = ax;
            ax = bx; Fax = Fbx;
            bx = a + g * (b - a);
            Fbx = F.object(bx);
        else
            b = bx;
            bx = ax; Fbx = Fax;
            ax = a + (1 - g)*(b - a);
            Fax = F.object(ax);
        end
    end
    
    if Fax > Fbx
        nx = bx; ny = Fbx;
    else
        nx = ax; ny = Fax;
    end
end


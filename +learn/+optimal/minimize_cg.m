function [x1,y1] = minimize_cg(F,x0,parameters)
%minimize_cg �����ݶȷ�
%   parameters.epsilon1 ���ݶ�ģС��epsilon1ʱֹͣ����
%   parameters.epsilon2 ����������ֹͣ����
%   parameters.max_it ����������
%   parameters.reset ��������
%   parameters.dis ����������������

    x1 = x0; y1 = F.object(x0); % ��ʼ��Ϊx0,�������ʼ��Ŀ�꺯��ֵ 
    g1 = F.gradient(x1); % ����x1�����ݶȣ���ʱx1=x0��
    d1 = -g1; % ��ʼ��������Ϊ���ݶȷ���
    ng = norm(g1); % �����ݶ�ģ
    if ng < parameters.epsilon1
        return;
    end
    
    alfa = learn.optimal.search(F,x1,d1,0,parameters.dis,parameters.epsilon2);
    
    for it = 1:parameters.max_it
        ng = norm(g1); % �����ݶ�ģ
        if ng < parameters.epsilon1
            break;
        end
        x2 = x1 + alfa * d1; 
        y2 = F.object(x2);
        if mod(it,parameters.reset) == 0 || y1 < y2
            d1 = -g1;
            alfa = learn.optimal.search(F,x1,d1,0,parameters.dis,parameters.epsilon2);
            disp(sprintf('Ŀ�꺯��:%f ��������:%d �ݶ�ģ:%f ',y1,it,ng));
            continue;
        end
        
        disp(sprintf('Ŀ�꺯��:%f ��������:%d �ݶ�ģ:%f ',y1,it,ng));
        g2 = F.gradient(x2); % ����x2�����ݶ�
        beda = g2'*(g2-g1)/(g1'*g1);
        d2 = -g2 + beda * d1;
        alfa = learn.optimal.search(F,x2,d2,0,parameters.dis,parameters.epsilon2);
        x1 = x2; d1 = d2; g1 = g2; y1 = y2;
    end
end


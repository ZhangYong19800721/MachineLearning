function [x1,y1] = minimize_cg(F,x0,parameters)
%minimize_cg �����ݶȷ�
% ���룺
%   F �������󣬵���F.object(x)����Ŀ�꺯����x����ֵ������F.gradient(x)����Ŀ�꺯����x�����ݶ�
%   x0 ��������ʼλ��
%   parameters.epsilon ���ݶ�ģС��epsilonʱֹͣ����
%   parameters.alfa alfa*epsilon������������ֹͣ����
%   parameters.max_it ����������
%   parameters.reset ��������
%   parameters.dis ����������������  

    x1 = x0; y1 = F.object(x0); % ��ʼ��Ϊx0,�������ʼ��Ŀ�꺯��ֵ 
    g1 = F.gradient(x1); % ����x1�����ݶȣ���ʱx1=x0��
    d1 = -g1; % ��ʼ��������Ϊ���ݶȷ���
    ng = norm(g1); % �����ݶ�ģ
    if ng < parameters.epsilon
        return;
    end
    
    alfa = learn.optimal.search(F,x1,d1,0,parameters.dis,parameters.alfa*parameters.epsilon);
    
    for it = 1:parameters.max_it
        ng = norm(g1); % �����ݶ�ģ
        if ng < parameters.epsilon
            break;
        end
        x2 = x1 + alfa * d1; 
        y2 = F.object(x2);
        if mod(it,parameters.reset) == 0 || y1 < y2
            d1 = -g1;
            alfa = learn.optimal.search(F,x1,d1,0,parameters.dis,parameters.alfa*parameters.epsilon);
            disp(sprintf('Ŀ�꺯��:%f ��������:%d �ݶ�ģ:%f ',y1,it,ng));
            continue;
        end
        
        disp(sprintf('Ŀ�꺯��:%f ��������:%d �ݶ�ģ:%f ',y1,it,ng));
        g2 = F.gradient(x2); % ����x2�����ݶ�
        beda = g2'*(g2-g1)/(g1'*g1);
        d2 = -g2 + beda * d1;
        alfa = learn.optimal.search(F,x2,d2,0,parameters.dis,parameters.alfa*parameters.epsilon);
        x1 = x2; d1 = d2; g1 = g2; y1 = y2;
    end
end


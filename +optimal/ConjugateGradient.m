function [x1,y1] = ConjugateGradient(F,x0,epsilon1,epsilon2,max_it,reset,dis)
%CONJUGATEGRADIENT �����ݶȷ�
%   
    ob = learn.Observer('Ŀ�꺯��ֵ',1,100,'xxx');
    x1 = x0; y1 = F.object(x0); % ��ʼ��Ϊx0,�������ʼ��Ŀ�꺯��ֵ 
    g1 = F.gradient(x1); % ����x1�����ݶȣ���ʱx1=x0��
    d1 = -g1; % ��ʼ��������Ϊ���ݶȷ���
    if norm(g1) < epsilon1
        return;
    end
    
    alfa = optimal.GoldenSection(F,x1,d1,0,dis,epsilon2);
    
    for it = 1:max_it
        if norm(g1) < epsilon1
            break;
        end
        x2 = x1 + alfa * d1; 
        y2 = F.object(x2);
        if mod(it,reset) == 0 || y1 < y2
            d1 = -g1;
            alfa = optimal.GoldenSection(F,x1,d1,0,dis,epsilon2);
            
            description = strcat(strcat(strcat('��������:',num2str(it)),'/'),num2str(max_it));
            description = strcat(description,strcat(' �ݶ�ģ:',num2str(norm(g1))));
            ob = ob.showit(y1,description);
            continue;
        end
        
        description = strcat(strcat(strcat('��������:',num2str(it)),'/'),num2str(max_it));
        description = strcat(description,strcat(' �ݶ�ģ:',num2str(norm(g1))));
        ob = ob.showit(y1,description);
        
        g2 = F.gradient(x2); % ����x2�����ݶ�
        beda = g2'*(g2-g1)/(g1'*g1);
        d2 = -g2 + beda * d1;
        alfa = optimal.GoldenSection(F,x2,d2,0,dis,epsilon2);
        x1 = x2; d1 = d2; g1 = g2; y1 = y2;
    end
end


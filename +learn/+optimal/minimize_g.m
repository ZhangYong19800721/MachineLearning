function [x,y] = minimize_g(F,x0,parameters)
%minimize_g �ݶȷ�
%   ���룺
%       F ����F.gradient(x)����Ŀ�꺯�����ݶȣ�����F.object(x)����Ŀ�꺯����ֵ
%       x0 ��������ʼλ��
%       parameters.learn_rate ѧϰ�ٶ�
%       parameters.momentum ���ٶ���
%       parameters.epsilon ���ݶȵķ���С��epsilonʱ��������
%       parameters.max_it ����������
%   �����
%       x ���ŵĲ�����
%       y ��С�ĺ���ֵ
    
    %% ��ʼ��
    ob = learn.tools.Observer('Ŀ�꺯��ֵ',1,100); % ��ʼ���۲���
    inc_x = zeros(size(x0)); % �����ĵ�����
    m = parameters.momentum;
    r = parameters.learn_rate;
    x1 = x0; 
    
    %% ��ʼ����
    for it = 1:parameters.max_it
        g1 = F.gradient(x1); % �����ݶ�
        y1 = F.object(x1); % ����Ŀ�꺯��ֵ
        ng = norm(g1); % �����ݶ�ģ
        
        description = sprintf('ѧϰ�ٶȣ�%f ��������: %d �ݶ�ģ��%f ',r,it,ng);
        ob = ob.showit(y1,description);
        
        if ng < parameters.epsilon
            break; % ����ݶ��㹻С�ͽ�������
        end
        
        inc_x = m * inc_x - (1 - m) * r * g1; % ���ݶȷ����������ʹ�ö�������
        x1 = x1 + inc_x; % ���²���ֵ
    end
    
    %% ����
    x = x1;
    y = y1;
end


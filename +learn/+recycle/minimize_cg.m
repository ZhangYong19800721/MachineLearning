function [x1,y1] = minimize_cg(F,x0,parameters)
%minimize_cg �����ݶȷ�
% ���룺
%   F �������󣬵���F.object(x)����Ŀ�꺯����x����ֵ������F.gradient(x)����Ŀ�꺯����x�����ݶ�
%   x0 ��������ʼλ��
%   parameters.epsilon ���ݶ�ģС��epsilonʱֹͣ����
%   parameters.alfa �����������䱶��
%   parameters.beda ����������ֹͣ����
%   parameters.max_it ����������
%   parameters.reset ��������

    %ob = learn.tools.Observer('����ֵ',1,100);
    %% ������ʼλ�õĺ���ֵ���ݶȡ��ݶ�ģ
    x1 = x0; y1 = F.object(x1); g1 = F.gradient(x1); ng1 = norm(g1); % ��ʼ��Ϊx0,���㺯��ֵ���ݶȡ��ݶ�ģ 
    if ng1 < parameters.epsilon, return; end % ����ݶ��㹻С��ֱ�ӷ���
    
    %% ����Ѱ��
    d1 = -g1; % ��ʼ��������Ϊ���ݶȷ���
    for it = 1:parameters.max_it
        if ng1 < parameters.epsilon, return; end % ����ݶ��㹻С��ֱ�ӷ���
        [~,y2,x2] = learn.optimal.armijo(F,x1,g1,parameters.alfa * d1,parameters);
%         alfa = learn.optimal.search(F,x1,d1,0,parameters.alfa,parameters.beda); % ��d1����������
%         x2 = x1 + alfa * d1; y2 = F.object(x2); % �������µ�λ��x2�������㺯��ֵ
        c1 = mod(it,parameters.reset) == 0; % �������õ�
        c2 = y1 < y2; %����d1������һ���½�����
        if c1 || c2
            d1 = -g1; % �趨��������Ϊ���ݶȷ���
            [~,y2,x2] = learn.optimal.armijo(F,x1,g1,parameters.alfa * d1,parameters); % �ظ��ݶȷ���������
            % alfa = learn.optimal.search(F,x1,d1,0,parameters.alfa,parameters.beda); 
            % x2 = x1 + alfa * d1; y2 = F.object(x2); 
            g2 = F.gradient(x2); d2 = -g2; ng2 = norm(g2); % �������µ�λ��x2���������ݶȡ����������ݶ�ģ
            x1 = x2; d1 = d2; g1 = g2; y1 = y2; ng1 = ng2;
            disp(sprintf('Ŀ�꺯��:%f ��������:%d �ݶ�ģ:%f ',y1,it,ng1));
            %ob = ob.showit(y1,'hello');
            continue;
        end
 
        g2 = F.gradient(x2); ng2 = norm(g2); % ����x2�����ݶȺ��ݶ�ģ
        beda = g2'*(g2-g1)/(g1'*g1); d2 = -g2 + beda * d1; % ����x2������������d2
        x1 = x2; d1 = d2; g1 = g2; y1 = y2; ng1 = ng2;
        disp(sprintf('Ŀ�꺯��:%f ��������:%d �ݶ�ģ:%f ',y1,it,ng1));
        %ob = ob.showit(y1,'hello');
    end
end


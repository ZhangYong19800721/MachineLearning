classdef Perception
    %PERCEPTION ��֪��
    %   
    
    properties
        weight; % Ԫ�����飬weight{m}��ʾ��m���Ȩֵ
        bias;   % Ԫ�����飬bias{m}  ��ʾ��m���ƫ��ֵ
    end
    
    methods % ���캯��
        function obj = Perception(configure,f)
            % Perception ���캯��
            M = length(configure) - 1;
            for m = 1:M
                obj.weight{m} = zeros(configure(m+1),configure(m));
                obj.bias{m}   = zeros(configure(m+1),1);
            end
        end
    end
    
    methods
        function obj = initialize(obj)
            M = length(obj.bias); % �õ�����
            for m = 1:M
                obj.weight{m} = 0.01 * randn(size(obj.weight{m})); % ��Ȩֵ��ʼ��Ϊ0�����������
                obj.bias{m} = zeros(size(obj.bias{m}));            % ��ƫ��ֵ��ʼ��Ϊ0
            end
        end
        
        function obj = train(obj,minibatchs,learn_rate_min,learn_rate_max,max_it)
            minibatch_num = length(minibatchs);
            layer_num = length(obj.bias);
            ob_window_size = minibatch_num;
            ob = ML.Observer('�������',1,ob_window_size,'xxx');
            
            % ��ʼ����������Ϊ0.5
            momentum = 0.5;
            for m = 1:layer_num
                velocity{m}.weight = zeros(size(obj.weight{m}));
                velocity{m}.bias   = zeros(size(obj.bias  {m}));
            end
            
            learn_rate = learn_rate_max; % ��ʼ��ѧϰ�ٶ�Ϊ���ѧϰ�ٶ�
            error_list = zeros(1,ob_window_size);
            for minibatch_idx = 1:minibatch_num  % ��ʼ������б���ƶ�ƽ��ֵ
                labels = minibatchs{minibatch_idx}.labels;
                points = minibatchs{minibatch_idx}.points;
                error = sum(sqrt(sum((labels - obj.do(points)).^2,2)));
                error_list(minibatch_idx) = error;
            end
            error_ave_old = mean(error_list);
            ob = ob.initialize(error_ave_old);
            
            for it = 0:max_it
                minibatch_idx = mod(it,minibatch_num) + 1;
                labels = minibatchs{minibatch_idx}.labels;
                points = minibatchs{minibatch_idx}.points; 
                minibatch_size = size(points,2);
                error = sum(sqrt(sum((labels - obj.do(points)).^2,2)));
                error_list(minibatch_idx) = error;
                error_ave_new = mean(error_list);
                
                if minibatch_idx == minibatch_num
                    if error_ave_new > error_ave_old
                        learn_rate = 0.5 * learn_rate;
                        if learn_rate < learn_rate_min
                            break;
                        end
                    end
                    error_ave_old = error_ave_new;
                end
                
                description = strcat(strcat(strcat('��������:',num2str(it)),'/'),num2str(max_it));
                description = strcat(description,strcat('ѧϰ�ٶ�:',num2str(learn_rate)));
                ob = ob.showit(error_ave_new,description);
                
                for m = 1:layer_num
                    delta{m}.weight = zeros(size(obj.weight{m}));
                    delta{m}.bias   = zeros(size(obj.bias  {m}));
                end
                
                for n = 1:minibatch_size
                    delta_n = obj.BP(points(:,n),labels(:,n));
                    for m = 1:layer_num
                        delta{m}.weight = delta{m}.weight + delta_n{m}.weight;
                        delta{m}.bias   = delta{m}.bias   + delta_n{m}.bias;
                    end
                end
                
                momentum = min([momentum * 1.01,0.9]); % �����������Ϊ0.9����ʼֵΪ0.5����Լ����60��֮�������ʴﵽ0.9��
                for m = 1:layer_num
                    velocity{m}.weight = momentum * velocity{m}.weight + learn_rate * delta{m}.weight / minibatch_size;
                    velocity{m}.bias   = momentum * velocity{m}.bias   + learn_rate * delta{m}.bias   / minibatch_size;
                end
                
                for m = 1:layer_num
                    obj.weight{m} = obj.weight{m} - velocity{m}.weight;
                    obj.bias{m}   = obj.bias{m}   - velocity{m}.bias;
                end
            end
        end
        
        function [y,n,a] = do(obj,x)
            % ����֪���ļ������
            % y ���������
            % n ��ÿһ��ľֲ��յ���
            % a ��ÿһ������
            
            M = length(obj.bias);           % �õ�����
            n = cell(1,M); a = cell(1,M);   % n��ÿһ��ľֲ��յ���a��ÿһ������
          
            for m = 1:M
                n{m} = obj.weight{m} * x + repmat(obj.bias{m},1,size(x,2));
                if m < M
                    a{m} = ML.sigmoid(n{m});
                else
                    a{m} = n{m};
                end
                x = a{m};
            end
            
            y = a{M};
        end
    end
    
    methods (Access = private)
        function delta = BP(obj,point,label)
            % ���򴫲��㷨
            M = length(obj.bias); % �õ�����

            [~,~,a] = obj.do(point); % ����ִ���������,����¼ÿһ������
           
            %s{M} = 2 * (label - a{M}) .* a{M} .* (a{M} - 1); % ���㶥���������
            s{M} = -2 * (label - a{M}); % ���㶥���������
            
            for m = (M-1):-1:1  % ���򴫲�������
                s{m} = diag(a{m} .* (1 - a{m})) * obj.weight{m+1}' * s{m+1}; 
            end
            
            for m = 1:M  % �����ݶ�
                if m == 1 
                    delta{m}.weight = repmat(s{m},1,size(point, 1)) .* repmat(point' ,size(a{m},1),1);
                    delta{m}.bias   = s{m};
                else
                    delta{m}.weight = repmat(s{m},1,size(a{m-1},1)) .* repmat(a{m-1}',size(a{m},1),1);
                    delta{m}.bias   = s{m};
                end
            end
        end
    end
    
    methods(Static)
        function [perception,e] = unit_test1()
            clear all;
            close all;
            [train_images,train_labels,test_images,test_labels] = ML.import_mnist('./+ML/mnist.mat');
            [D,minibatch_size,minibatch_num] = size(train_images); K = 10;
            for minibatch_idx = 1:minibatch_num
                mnist{minibatch_idx}.labels = zeros(K,minibatch_size);
                I = sub2ind([K,minibatch_size],1+train_labels(:,minibatch_idx),[1:minibatch_size]');
                mnist{minibatch_idx}.labels(I) = 1;
                mnist{minibatch_idx}.points = train_images(:,:,minibatch_idx);
            end
            
            configure = [D,500,600,2000,K];
            
            perception = ML.Perception(configure);
            perception = perception.initialize();
            perception = perception.train(mnist,1e-6,0.1,1e6);
            
            save('perception.mat','perception');
        end
        
        function [perception,e] = unit_test2()
            clear all;
            close all;
            daata = 100 * rand(2,1e4);
            label = daata(1,:) > daata(2,:);
            [D,N] = size(daata); K = 1;
            
            for minibatch_idx = 1:100
                data{minibatch_idx}.labels = label(:,(100 * (minibatch_idx - 1) + 1):(100 * (minibatch_idx - 1) + 100));
                data{minibatch_idx}.points = daata(:,(100 * (minibatch_idx - 1) + 1):(100 * (minibatch_idx - 1) + 100));
            end
            
            configure = [D,K];
            
            perception = ML.Perception(configure);
            perception = perception.initialize();
            perception = perception.train(data,1e-6,0.1,2e4);
            
            save('perception.mat','perception');
            
            y = perception.do(daata) > 0.5;
            e = sum(xor(y,label)) / length(y);
        end
        
        function [perception,e] = unit_test3()
            clear all;
            close all;
            M = 100; S = 100; N = M * S;
            r1 = 2 * rand(1,N) + 8; a1 =  pi * rand(1,N); group1 = repmat(r1,2,1) .* [cos(a1); sin(a1)];
            r2 = 2 * rand(1,N) + 8; a2 = -pi * rand(1,N); group2 = repmat(r2,2,1) .* [cos(a2); sin(a2)];
            group1(1,:) = group1(1,:) + 4; group1(2,:) = group1(2,:) - 2;   
            group2(1,:) = group2(1,:) - 4; group2(2,:) = group2(2,:) + 2; 
            figure(1);
            plot(group1(1,:),group1(2,:),'+'); hold on;
            plot(group2(1,:),group2(2,:),'o'); hold off;
            
            for minibatch_idx = 1:M
                data{minibatch_idx}.labels = [ones(1,S) zeros(1,S)];
                data{minibatch_idx}.points = [group1(:,(S*(minibatch_idx-1)+1):(S*(minibatch_idx-1)+S)) ...
                                              group2(:,(S*(minibatch_idx-1)+1):(S*(minibatch_idx-1)+S))];
            end
            
            configure = [2,20,1];
            perception = ML.Perception(configure);
            perception = perception.initialize();
            perception = perception.train(data,1e-6,0.1,1e5);
            
            save('perception.mat','perception');
            
            daata = [group1 group2];
            label = [ones(1,N) zeros(1,N)];
            y = perception.do(daata) > 0.5;
            e = sum(xor(y,label)) / length(y);
        end
        
        function [perception,e] = unit_test4()
            clear all;
            close all;
            M = 100; S = 100; N = M * S;
            x = linspace(-2,2,N); 
            k = 3;
            f = @(x)sin(k * pi * x / 4);
            l = f(x);
            
            configure = [1,6,1]; 
            perception = ML.Perception(configure);
            perception = perception.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,perception.do(x)); hold off;
            
            for minibatch_idx = 1:M
                data{minibatch_idx}.points = x(minibatch_idx:S:(M*S));
                data{minibatch_idx}.labels = f(data{minibatch_idx}.points);
            end
            
            perception = perception.train(data,1e-6,0.1,1e5);
            
            figure(3);
            y = perception.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            e = norm(l - y,2);
        end
    end
end


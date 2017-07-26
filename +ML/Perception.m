classdef Perception
    %PERCEPTION ��֪��
    %   
    
    properties
        weight; % Ԫ�����飬weight{m}��ʾ��m���Ȩֵ
        bias;   % Ԫ�����飬bias{m}��ʾ��m���ƫ��ֵ
    end
    
    methods % ���캯��
        function obj = Perception(configure)
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
            ob_window_size = minibatch_num;
            ob = ML.Observer('�������',1,ob_window_size,'xxx');
            
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
                % minibatch_size = size(points,2);
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
                
                delta = obj.BP(points(:,1),labels(:,1));
%                 delta_w = 2 * delta * points' / minibatch_size;
%                 delta_b = 2 * delta * ones(minibatch_size,1) / minibatch_size;
%                 obj.weight = obj.weight + learn_rate * delta_w;
%                 obj.bias   = obj.bias   + learn_rate * delta_b;
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
                a{m} = ML.sigmoid(n{m});
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
            s{M} = diag(2 * (label - a{M}) .* a{M} .* (a{M} - 1)); % ���㶥���������
            
            for m = (M-1):-1:1  % ���򴫲�������
                s{m} = s{m+1} * obj.weight{m+1} * diag(a{m} .* (1 - a{m})); 
            end
            
            for m = 1:M  % �����ݶ�
                if m == 1 
                    delta{m}.weight = s{m} * repmat(point',size(a{m}));
                    delta{m}.bias = 1;
                else
                    delta{m}.weight = s{m} * repmat(a{m-1}',size(a{m-1},1),1);
                    delta{m}.bias = 1;
                end
            end
        end
    end
    
    methods(Static)
        function [perception,e] = unit_test()
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
            
%             y = perception.classify(test_images);
%             e = sum(y~=test_labels') / length(y);
        end
    end
end


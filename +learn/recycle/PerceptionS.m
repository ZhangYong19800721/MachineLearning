classdef PerceptionS
    %PERCEPTIONS ��֪��
    %  ���һ�����Ϊsigmoid��Ԫ
    
    properties
        weight;      % һά���飬���в��Ȩֵ��ƫ��ֵ�����������һά������[P,1]
        num_hidden;  % num_hidden{m}��ʾ��m�������Ԫ����
        num_visual;  % num_visual{m}��ʾ��m�������Ԫ����
        star_w_idx;  % star_w_idx{m}��ʾ��m���Ȩֵ����ʼλ��
        stop_w_idx;  % stop_w_idx{m}��ʾ��m���Ȩֵ�Ľ���λ��
        star_b_idx;  % star_b_idx{m}��ʾ��m���ƫ�õ���ʼλ��
        stop_b_idx;  % stop_b_idx{m}��ʾ��m���ƫ�õĽ���λ��
    end
    
    methods % ���캯��
        function obj = PerceptionS(configure)
            % Perception ���캯��
            M = length(configure) - 1;
            obj.num_hidden{1} = configure(2);
            obj.num_visual{1} = configure(1);
            obj.star_w_idx{1} = 1;
            obj.stop_w_idx{1} = obj.num_hidden{1} * obj.num_visual{1};
            obj.star_b_idx{1} = obj.stop_w_idx{1} + 1;
            obj.stop_b_idx{1} = obj.stop_w_idx{1} + obj.num_hidden{1};
            
            for m = 2:M
                obj.num_hidden{m} = configure(m+1);
                obj.num_visual{m} = configure(m+0);
                obj.star_w_idx{m} = obj.stop_b_idx{m-1} + 1;
                obj.stop_w_idx{m} = obj.stop_b_idx{m-1} + obj.num_hidden{m} * obj.num_visual{m};
                obj.star_b_idx{m} = obj.stop_w_idx{m} + 1;
                obj.stop_b_idx{m} = obj.stop_w_idx{m} + obj.num_hidden{m};
            end
        end
    end
    
    methods
        function obj = initialize(obj)
            M = length(obj.num_hidden); % �õ�����
            for m = 1:M
                obj.weight(obj.star_w_idx{m}:obj.stop_w_idx{m},1) = ...
                    0.01 * randn(size([obj.star_w_idx{m}:obj.stop_w_idx{m}]')); % ��Ȩֵ��ʼ��Ϊ0�����������
                obj.weight(obj.star_b_idx{m}:obj.stop_b_idx{m},1) = ...
                    zeros(size([obj.star_b_idx{m}:obj.stop_b_idx{m}]')); % ��ƫ��ֵ��ʼ��Ϊ0
            end
        end
        
        function [y,a] = do(obj,x)
            % ����֪���ļ������
            % y ���������
            
            M = length(obj.num_hidden); % �õ�����
            a = cell(1,M);          
            for m = 1:M
                w = obj.getw(m); b = obj.getb(m);
                a{m} = learn.tools.sigmoid(w * x + repmat(b,1,size(x,2)));
                x = a{m};
            end
            
            y = a{M};
        end
        
        function [w,r] = getw(obj,m)
            r = obj.star_w_idx{m}:obj.stop_w_idx{m};
            w = reshape(obj.weight(r),obj.num_hidden{m},obj.num_visual{m});
        end
        
        function [b,r] = getb(obj,m)
            r = obj.star_b_idx{m}:obj.stop_b_idx{m};
            b = reshape(obj.weight(r),[],1);
        end
    end
    
    methods(Static)
        function ps = unit_test()
            clear all;
            close all;
            rng(1);
            
            [mnist,~,~,~] = learn.import_mnist('./+learn/mnist.mat');
            [D,S,M] = size(mnist); mnist = reshape(mnist,D,[]); N = S*M;
            
            configure = [D,500,256,500,D];
            ps = learn.PerceptionS(configure);
            
            load('sae_mnist_finetune.mat');
            %load('sae_mnist_pretrain.mat');
            
            ps.weight = [reshape(sae.rbms{1}.weight_v2h,[],1);
                reshape(sae.rbms{1}.hidden_bias,[],1);
                reshape(sae.rbms{2}.weight_v2h,[],1);
                reshape(sae.rbms{2}.hidden_bias,[],1);
                reshape(sae.rbms{2}.weight_h2v',[],1);
                reshape(sae.rbms{2}.visual_bias,[],1);
                reshape(sae.rbms{1}.weight_h2v',[],1);
                reshape(sae.rbms{1}.visual_bias,[],1)];
           
            
            recon_mnist = ps.do(mnist);
            e1 = sum(sum((recon_mnist - mnist).^2)) / N
            
            cgbps = learn.CGBPS(mnist,mnist,ps);

            weight = optimal.minimize_cg(cgbps,ps.weight,1e-6,1e-3,1e6,inf,10);
            ps.weight = weight;
            
            recon_mnist = ps.do(mnist);
            e2 = sum(sum((recon_mnist - mnist).^2)) / N
        end
    end
end
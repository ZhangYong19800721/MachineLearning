classdef Observer
    %OBSERVER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        handle;
        window;
        legend;
        index;
        list;
        num;
    end
    
    methods
        function obj = Observer(name,num,window,legend)
            obj.handle = figure('Name',name); grid on;
            obj.window = window;
            obj.legend = legend;
            obj.num = num;
            obj.list = zeros(obj.num,obj.window);
            obj.index = 0;
        end
        
        function obj = showit(obj,data,description) % 在图中显示新给出的数据
            obj.index = mod(obj.index,obj.window);
            obj.list(:,obj.index+1) = data;
            
            figure(obj.handle);
            
            switch obj.num
                case 1
                    plot(1:obj.window,obj.list(1,:));
                case 2
                    plot(1:obj.window,obj.list(1,:),1:obj.window,obj.list(2,:));
                case 3
                    plot(1:obj.window,obj.list(1,:),1:obj.window,obj.list(2,:),1:obj.window,obj.list(3,:));
                case 4
                    plot(1:obj.window,obj.list(1,:),1:obj.window,obj.list(2,:),1:obj.window,obj.list(3,:),1:obj.window,obj.list(4,:));
            end
            grid on;
            title(description);
            xlim([0 obj.window]);
            obj.index = obj.index + 1;
            drawnow
        end
        
        function obj = initialize(obj,data)
            %INIT 初始化图中的初始数据
            %   此处显示详细说明
            obj.list = repmat(data,1,obj.window);
        end
    end
end


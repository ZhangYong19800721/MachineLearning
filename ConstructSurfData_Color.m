function SURF = ConstructSurfData_Color(image_base_dir,file_type)
%CONSTRUCTSURFDATA 从给定的目录中读取所有指定类型图片的SURF特征
%   SURF特征集合
    message = 'constructing surf data ....'; disp(message);
    files = dir(fullfile(image_base_dir,file_type)); % 得到所有图片的文件名
    N = length(files); % 图片文件的个数
    SURF = [];
    
    for n = 1:N
        read_image_file_number = n
        image_rgb = imread(strcat(image_base_dir,files(n).name));
        [~,width,~] = size(image_rgb);
        image_rgb_scale_down = imresize(image_rgb,320/width);
        image_yuv = rgb2ycbcr(image_rgb_scale_down);
        points_U = detectSURFFeatures(image_yuv(:,:,2));
        points_V = detectSURFFeatures(image_yuv(:,:,3));
        [features_U,~] = extractFeatures(image_yuv(:,:,2),points_U.selectStrongest(450));
        [features_V,~] = extractFeatures(image_yuv(:,:,3),points_V.selectStrongest(450));
        SURF = [SURF features_U' features_V'];
    end
end


function img = border_filler(img,width,mode)
%此函数用于对图像进行边界填充，img是待填充图像，width是填充宽度，mode有两个取值，
% 'zero'是零填充，'copy'是复制边缘填充
    [m,n] = size(img);%获取图像宽度
    if strcmp(mode,'zero')%零填充情况
        for i = 1:1:width%每次循环填充一周
            img = [zeros(1,n);img;zeros(1,n)];%纵方向扩充
            img = [zeros(m+2,1),img,zeros(m+2,1)];%横方向扩充
        end
    elseif strcmp(mode,'copy')%复制边缘扩充情况
        for i = 1:1:width%每次循环填充一周
            img = [img(1,:);img;img(end,:)];%纵方向扩充
            img = [img(:,1),img,img(:,end)];%横方向扩充
        end
    end
end
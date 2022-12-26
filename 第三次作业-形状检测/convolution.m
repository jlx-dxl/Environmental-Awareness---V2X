function img = convolution(M,filter)
%卷积函数，M为待处理图像，filter为掩膜
    [m,n] = size(filter);
    x_width = (m-1)/2;%得到掩膜x方向宽度
    y_width = (n-1)/2;%得到掩膜y方向宽度
    [m,n] = size(M);
    img = zeros(m-2*x_width,n-2*y_width);%卷积后图像宽度减小，得到结果存放矩阵
    M = double(M);%转换为浮点便于运算
    %卷积
    for i = -x_width:1:x_width
        for j = -y_width:1:y_width
            img = img + M(x_width+1+i:m-x_width+i,y_width+1+j:n-y_width+j)*filter(i+x_width+1,j+y_width+1); 
        end
    end
    
end
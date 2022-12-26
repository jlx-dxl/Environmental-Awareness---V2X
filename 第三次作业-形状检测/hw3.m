%% 数据准备
lanes = imread('..\source_images\lanes.png');%读取车道线图像
wheels = imread('..\source_images\wheel.png');%读取车轮图像
lanes = rgb2gray(lanes);%转灰度图
wheels = rgb2gray(wheels);%转灰度图
width = 2;%高斯滤波器宽度
ind = -width:1:width;%下面几步用于生成高斯掩膜
sigma = 1;%高斯滤波器方差
sgauss = 1/(sqrt(2*pi)*sigma)*exp(-ind.^2/(2*sigma^2));%一维高斯滤波器
gauss = sgauss'*sgauss;%二维高斯滤波器
gauss = gauss/sum(gauss(:));%归一化
xsobel = 1/8*[-1,0,1;-2,0,2;-1,0,1];%xsobel滤波器
ysobel = xsobel';%ysobel滤波器
%% canny变换
gauss_lanes = convolution(lanes,gauss);%车道线图像高斯滤波后
gx_lanes = convolution(gauss_lanes,xsobel);%x方向梯度
gy_lanes = -convolution(gauss_lanes,ysobel);%y方向梯度
%gx_lanes = convolution(lanes,xsobel);%x方向梯度
%gy_lanes = convolution(lanes,ysobel);%y方向梯度
g_lanes = sqrt(gx_lanes.^2+gy_lanes.^2);%梯度幅值
imwrite(uint8(255*g_lanes/max(g_lanes(:))),'..\result\sobel_origin_GM_Map.png')
theta_lanes = atan2(gy_lanes,gx_lanes);%梯度角度
[m,n] = size(theta_lanes);
theta_lanes(theta_lanes>pi/2) = theta_lanes(theta_lanes>pi/2)-pi;%通过下面两步将theta矩阵映射至-pi/2，pi/2的区间中，便于后续计算
theta_lanes(theta_lanes<=-pi/2) = theta_lanes(theta_lanes<=-pi/2) + pi;
t_theta = -theta_lanes + pi/2;
ind_theta = floor(t_theta/pi*4)+1;%将theta分为四个区间，对应如下
%   值   区间
%   4   (-pi/2,-pi/4]
%   3   (-pi/4,0]
%   2   (0,pi/4]
%   1   (pi/4,pi/2]
%每一个theta需要同时比对正负梯度方向上的梯度值，对应两组梯度值及两组系数坐标，以落
%在(-pi/2,-pi/4]的点为例，这部分点需要用S和ES、N和WN方向的像素点梯度值进行插值，
%将其存储在G1-G4中，总是保证系数矩阵和梯度矩阵的位置对应，这样可以节省一组系数坐标
%   value   4   3   2   1
%   G1      S   SE  E   NE
%   G2      SE  E   NE  N
%   G3      N   NW  W   SW
%   G4      NW  W   SW  S
%   c1      1-|1/tanθ|  |tanθ|      1-|tanθ|    |1/tanθ|
%   c2      |1/tanθ|    1-|tanθ|    |tanθ|      1-|1/tanθ|
%GP1 = c1*G1+c2*G2  GP2 = c1*G3+c2*G4   c2 = 1-c1
G1 = zeros(m,n);
G2 = zeros(m,n);
G3 = zeros(m,n);
G4 = zeros(m,n);
c1 = zeros(m,n);
c2 = zeros(m,n);
GP1 = zeros(m,n);
GP2 = zeros(m,n);
ex1_g_lanes = border_filler(g_lanes,1,'copy');%将图像扩充一格，解决边缘只有一个梯度的问题
%通过遍历索引矩阵对上述六个矩阵赋值，并计算GP1、GP2矩阵
for i = 1:1:m
    for j = 1:1:n
        if ind_theta(i,j) == 4
            G1(i,j) = ex1_g_lanes(i+1+1,j+1);%S
            G2(i,j) = ex1_g_lanes(i+1+1,j+1+1);%SE
            G3(i,j) = ex1_g_lanes(i+1-1,j+1);%N
            G4(i,j) = ex1_g_lanes(i+1-1,j+1-1);%NW
            c1(i,j) = 1-1/abs(tan(theta_lanes(i,j)));%1-|1/tanθ|
        elseif ind_theta(i,j) == 3
            G1(i,j) = ex1_g_lanes(i+1+1,j+1+1);%SE
            G2(i,j) = ex1_g_lanes(i+1,j+1+1);%E
            G3(i,j) = ex1_g_lanes(i+1-1,j+1-1);%NW
            G4(i,j) = ex1_g_lanes(i+1,j+1-1);%W
            c1(i,j) = abs(tan(theta_lanes(i,j)));%|tanθ|
        elseif ind_theta(i,j) == 2
            G1(i,j) = ex1_g_lanes(i+1,j+1+1);%E
            G2(i,j) = ex1_g_lanes(i+1-1,j+1+1);%NE
            G3(i,j) = ex1_g_lanes(i+1,j+1-1);%W
            G4(i,j) = ex1_g_lanes(i+1+1,j+1-1);%SW
            c1(i,j) = 1-abs(tan(theta_lanes(i,j)));%1-|tanθ|
        elseif ind_theta(i,j) == 1
            G1(i,j) = ex1_g_lanes(i+1-1,j+1+1);%NE
            G2(i,j) = ex1_g_lanes(i+1-1,j+1);%N
            G3(i,j) = ex1_g_lanes(i+1+1,j+1-1);%SW
            G4(i,j) = ex1_g_lanes(i+1+1,j+1);%S
            c1(i,j) = 1/abs(tan(theta_lanes(i,j)));%|1/tanθ|
        end
    end
end
c2 = 1-c1;%计算c2矩阵
GP1 = c1.*G1+c2.*G2;%计算GP1
GP2 = c1.*G3+c2.*G4;%计算GP2
com_g_GP1 = g_lanes>=GP1;%将原始梯度矩阵与GP1对比
com_g_GP2 = g_lanes>=GP2;%将原始梯度矩阵与GP2对比
nms_g_lanes = g_lanes.*com_g_GP1.*com_g_GP2;%仅当原始梯度最大时保留梯度值
imwrite(uint8(255*nms_g_lanes/max(nms_g_lanes(:))),'..\result\NMS_GM_Map.png')
lowerlimit = 14.6;%下限
higherlimit = 28;%上限
s_border = nms_g_lanes>=higherlimit;%强边缘
w_border = (lowerlimit<=nms_g_lanes)&(nms_g_lanes<higherlimit);%弱边缘

state = zeros(m,n);%状态矩阵
%为了便于处理边缘像素，将所有矩阵零填充一格
ex1_s_border = border_filler(s_border,1,'zero');
ex1_w_border = border_filler(w_border,1,'zero');
ex1_state = border_filler(state,1,'zero');
[m,n] = size(ex1_s_border);
wbs = find(ex1_w_border==1);
for i = 1:1:length(wbs)
    cp = wbs(i);%取出一个弱边缘点
    if ex1_state(cp)==0%只有当该点没被搜索过时才搜索该点
        ex1_state(cp)=1;%改变状态
        currentp = [cp];%搜索集
        resultp = [cp];%结果集
        flag = 0;
        while ~isempty(currentp)
            [r,c] = ind2sub(size(ex1_s_border),currentp(1));%将索引转下标，此时下标不会位于边缘处
            
            ex1_state(cp)=1;%改变状态
            currentp(1) = [];%第一个点被搜索过，将其移出搜索集
            rs = [r-1,r-1,r-1,r,r,r+1,r+1,r+1];%周围八个点的行坐标
            cs = [c-1,c,c+1,c-1,c+1,c-1,c,c+1];%周围八个点的列坐标
            inds = sub2ind([m,n],rs,cs);%坐标转索引
            state_s = ex1_state(inds);%获取这些点的状态
            w_s = ex1_w_border(inds).*(1-state_s);%只有当状态为0时结果才有效
            s_s = ex1_s_border(inds);%搜索是否与强边界联通
            if sum(s_s)>0%如果至少一个点与强边界联通，flag置1
                flag = 1;
            end
            currentp = [currentp,inds(w_s==1)];%将所有满足条件的弱边界点纳入搜索集
            ex1_state(inds(w_s==1)) = 1;%将所有满足条件的弱边界点状态置为1
            resultp = [resultp,inds(w_s==1)];%将所有弱边界点集合
        end
        if flag == 1%当弱边界与强边界有交点时置为强边界
            ex1_s_border(resultp) = 1;
        end
    end

end
b_result = ex1_s_border(2:end-1,2:end-1);%二值图
dt_result = b_result.*nms_g_lanes;%双阈值处理后的图
imwrite(uint8(255*dt_result/max(dt_result(:))),'..\result\DT_GM_Map.png')%输出双阈值处理后的图
imwrite(uint8(255*b_result/max(b_result(:))),'..\result\Binary_GM_Map.png')%输出二值化后的图
%subplot(211)
%imshow(nms_g_lanes,[])
%subplot(212)
%imshow(b_result,[])

%% 霍夫变换 对车轮canny变换
gauss_wheels = convolution(wheels,gauss);%车轮图像高斯滤波后
gx_wheels = convolution(gauss_wheels,xsobel);%x方向梯度
gy_wheels = -convolution(gauss_wheels,ysobel);%y方向梯度

g_wheels = sqrt(gx_wheels.^2+gy_wheels.^2);%梯度幅值
theta_wheels = atan2(gy_wheels,gx_wheels);%梯度角度
[m,n] = size(theta_wheels);
theta_wheels(theta_wheels>pi/2) = theta_wheels(theta_wheels>pi/2)-pi;%通过下面两步将theta矩阵映射至-pi/2，pi/2的区间中，便于后续计算
theta_wheels(theta_wheels<=-pi/2) = theta_wheels(theta_wheels<=-pi/2) + pi;
t_theta = -theta_wheels + pi/2;
ind_theta = floor(t_theta/pi*4)+1;%将theta分为四个区间，对应如下
%   值   区间
%   4   (-pi/2,-pi/4]
%   3   (-pi/4,0]
%   2   (0,pi/4]
%   1   (pi/4,pi/2]
%每一个theta需要同时比对正负梯度方向上的梯度值，对应两组梯度值及两组系数坐标，以落
%在(-pi/2,-pi/4]的点为例，这部分点需要用S和ES、N和WN方向的像素点梯度值进行插值，
%将其存储在G1-G4中，总是保证系数矩阵和梯度矩阵的位置对应，这样可以节省一组系数坐标
%   value   4   3   2   1
%   G1      S   SE  E   NE
%   G2      SE  E   NE  N
%   G3      N   NW  W   SW
%   G4      NW  W   SW  S
%   c1      1-|1/tanθ|  |tanθ|      1-|tanθ|    |1/tanθ|
%   c2      |1/tanθ|    1-|tanθ|    |tanθ|      1-|1/tanθ|
%GP1 = c1*G1+c2*G2  GP2 = c1*G3+c2*G4   c2 = 1-c1
G1 = zeros(m,n);
G2 = zeros(m,n);
G3 = zeros(m,n);
G4 = zeros(m,n);
c1 = zeros(m,n);
c2 = zeros(m,n);
GP1 = zeros(m,n);
GP2 = zeros(m,n);
ex1_g_wheels = border_filler(g_wheels,1,'copy');%将图像扩充一格，解决边缘只有一个梯度的问题
%通过遍历索引矩阵对上述六个矩阵赋值，并计算GP1、GP2矩阵
for i = 1:1:m
    for j = 1:1:n
        if ind_theta(i,j) == 4
            G1(i,j) = ex1_g_wheels(i+1+1,j+1);%S
            G2(i,j) = ex1_g_wheels(i+1+1,j+1+1);%SE
            G3(i,j) = ex1_g_wheels(i+1-1,j+1);%N
            G4(i,j) = ex1_g_wheels(i+1-1,j+1-1);%NW
            c1(i,j) = 1-1/abs(tan(theta_wheels(i,j)));%1-|1/tanθ|
        elseif ind_theta(i,j) == 3
            G1(i,j) = ex1_g_wheels(i+1+1,j+1+1);%SE
            G2(i,j) = ex1_g_wheels(i+1,j+1+1);%E
            G3(i,j) = ex1_g_wheels(i+1-1,j+1-1);%NW
            G4(i,j) = ex1_g_wheels(i+1,j+1-1);%W
            c1(i,j) = abs(tan(theta_wheels(i,j)));%|tanθ|
        elseif ind_theta(i,j) == 2
            G1(i,j) = ex1_g_wheels(i+1,j+1+1);%E
            G2(i,j) = ex1_g_wheels(i+1-1,j+1+1);%NE
            G3(i,j) = ex1_g_wheels(i+1,j+1-1);%W
            G4(i,j) = ex1_g_wheels(i+1+1,j+1-1);%SW
            c1(i,j) = 1-abs(tan(theta_wheels(i,j)));%1-|tanθ|
        elseif ind_theta(i,j) == 1
            G1(i,j) = ex1_g_wheels(i+1-1,j+1+1);%NE
            G2(i,j) = ex1_g_wheels(i+1-1,j+1);%N
            G3(i,j) = ex1_g_wheels(i+1+1,j+1-1);%SW
            G4(i,j) = ex1_g_wheels(i+1+1,j+1);%S
            c1(i,j) = 1/abs(tan(theta_wheels(i,j)));%|1/tanθ|
        end
    end
end
c2 = 1-c1;%计算c2矩阵
GP1 = c1.*G1+c2.*G2;%计算GP1
GP2 = c1.*G3+c2.*G4;%计算GP2
com_g_GP1 = g_wheels>=GP1;%将原始梯度矩阵与GP1对比
com_g_GP2 = g_wheels>=GP2;%将原始梯度矩阵与GP2对比
nms_g_wheels = g_wheels.*com_g_GP1.*com_g_GP2;%仅当原始梯度最大时保留梯度值

lowerlimit = 7;%下限
higherlimit = 14.5;%上限
s_border = nms_g_wheels>=higherlimit;%强边缘
w_border = (lowerlimit<=nms_g_wheels)&(nms_g_wheels<higherlimit);%弱边缘

state = zeros(m,n);%状态矩阵
%为了便于处理边缘像素，将所有矩阵零填充一格
ex1_s_border = border_filler(s_border,1,'zero');
ex1_w_border = border_filler(w_border,1,'zero');
ex1_state = border_filler(state,1,'zero');
[m,n] = size(ex1_s_border);
wbs = find(ex1_w_border==1);
for i = 1:1:length(wbs)
    cp = wbs(i);%取出一个弱边缘点
    if ex1_state(cp)==0%只有当该点没被搜索过时才搜索该点
        ex1_state(cp)=1;%改变状态
        currentp = [cp];%搜索集
        resultp = [cp];%结果集
        flag = 0;
        while ~isempty(currentp)
            [r,c] = ind2sub(size(ex1_s_border),currentp(1));%将索引转下标，此时下标不会位于边缘处
            
            ex1_state(cp)=1;%改变状态
            currentp(1) = [];%第一个点被搜索过，将其移出搜索集
            rs = [r-1,r-1,r-1,r,r,r+1,r+1,r+1];%周围八个点的行坐标
            cs = [c-1,c,c+1,c-1,c+1,c-1,c,c+1];%周围八个点的列坐标
            inds = sub2ind([m,n],rs,cs);%坐标转索引
            state_s = ex1_state(inds);%获取这些点的状态
            w_s = ex1_w_border(inds).*(1-state_s);%只有当状态为0时结果才有效
            s_s = ex1_s_border(inds);%搜索是否与强边界联通
            if sum(s_s)>0%如果至少一个点与强边界联通，flag置1
                flag = 1;
            end
            currentp = [currentp,inds(w_s==1)];%将所有满足条件的弱边界点纳入搜索集
            ex1_state(inds(w_s==1)) = 1;%将所有满足条件的弱边界点状态置为1
            resultp = [resultp,inds(w_s==1)];%将所有弱边界点集合
        end
        if flag == 1%当弱边界与强边界有交点时置为强边界
            ex1_s_border(resultp) = 1;
        end
    end

end
b_result = ex1_s_border(2:end-1,2:end-1);%二值图
imwrite(b_result,'..\result\Binary_wheels.png')%为便于车轮检验，设定阈值的原则为保证车轮尽可能完整的同时减少其他轮廓
borders = find(b_result == 1);%获得所有强边缘
ts = theta_wheels(borders);%得到所有边缘点的倾斜角
[m,n] = size(b_result);%得到边缘矩阵大小
[bu,bv] = ind2sub([m,n],borders);%得到强边缘坐标 
%% 投票
pspace = zeros(m,n);%创建参数空间
v = 1:1:n;%创建赋值用序列
%将u,v坐标系关于x轴翻转，此时得到的xy坐标系下每一点的投票线为u=-tanθ(v-v0)+u0，只
%需将赋值序列代入后四舍五入即可得到对应参数空间内的票数，需要对u进行判断看是否可以
%赋值，由于角度取值范围为(-pi/2,pi/2]，需要特殊处理斜率不存在的点
for i = 1:1:length(borders)
    if ts(i) ~= pi/2
        u = -tan(ts(i))*(v-bv(i))+bu(i);%进行像素计算
        u = round(u);%四舍五入以转换到像素坐标系
        vv = find(1<=u & u<=m);%取出合理值
        vind = sub2ind([m,n],u(vv),v(vv));%将坐标转索引
    else
        u = 1:1:m;%当斜率不存在，对应像素点同一列全部加一
        vind = sub2ind([m,n],u,ones(1,m)*bv(i));
    end
    pspace(vind) = pspace(vind) + 1;
end
imwrite(uint8(255*pspace/max(pspace(:))),'..\result\Parameter_space.png')
%% 寻找圆心及半径
pspace = convolution(pspace,ones(7));%均值滤波防止奇异值影响
[u1,v1] = ind2sub(size(pspace),find(pspace == max(pspace(:))));%寻找最大值所在
p1 = [u1,v1]+6;%第一个圆心
%圆心需要转换到原图上，由于经历了一次5*5高斯，3*3sobel，7*7均值，需要加6
pspace(u1-5:u1+5,v1-5:v1+5) = 0;%将圆心周围点置0便于寻找第二圆心
[u2,v2] = ind2sub(size(pspace),find(pspace == max(pspace(:))));%寻找最大值所在
p2 = [u2,v2]+6;%第二个圆心
inds = find(b_result==1);%获得所有强边界点的索引
[x,y] = ind2sub(size(b_result),inds);%获得强边界点坐标
rs = sqrt((x-u1).^2+(y-v1).^2);%获得半径
rs = round(rs);%四舍五入便于统计半径
rspace = zeros(1,max(rs));%创建参数空间
for i = 1:1:length(rs)
    rspace(rs(i)) = rspace(rs(i))+1;%投票
end
rspace = rspace(20:50);
r1 = find(rspace == max(rspace(:)))+20;%半径

rs = sqrt((x-u2).^2+(y-v2).^2);%获得半径，对第二个点
rs = round(rs);%四舍五入便于统计半径
rspace = zeros(1,max(rs));%创建参数空间
for i = 1:1:length(rs)
    rspace(rs(i)) = rspace(rs(i))+1;%投票
end
rspace = rspace(15:50);%预先判断半径的合理范围
r2 = find(rspace == max(rspace(:)))+15;%半径
%% 绘图
wheels = imread('..\source_images\wheel.png');%读取车轮图像
fig = imshow(wheels);
axis xy
p1 = p1+0.5;%转直角坐标
p2 = p2+0.5;%转直角坐标
t=0:0.1:2*pi;%圆的坐标
x1 = p1(2)+r1*cos(t);
y1 = p1(1)+r1*sin(t);
x2 = p2(2)+r2*cos(t);
y2 = p2(1)+r2*sin(t);
hold on
plot(x1,y1,'r')
hold on 
plot(x2,y2,'r')
axis ij
axis image
saveas(fig,'..\result\Wheel_detect.png')














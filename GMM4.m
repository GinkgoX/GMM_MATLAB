function GMM8(path, set, type)
% 读取数据集图片列表
fileExt = '*.jpeg';
files = dir(fullfile(path, set, '\', type, '\', fileExt));
% 遍历数据集中的图片8高斯分量聚类
for i = 1 : 1 : size(files,1)
    if i == 1
        fid = fopen(strcat(path,set,'\',type,'\GMM4.csv'),'w');         % 第一张图片数据以新建或覆盖形式写入GMM8.csv
    else
        fid = fopen(strcat(path,set,'\',type,'\GMM4.csv'),'a');         % 后续图片数据以追加形式写入GMM8.csv
        fprintf(fid,'\n');
    end
    fprintf('%s\n',strcat(path,type,files(i).name));                    % 用于查看程序执行状态(显示文件名),可以注释掉
    Des = Core(i, strcat(path,set,'\',type,'\',files(i).name));         % 执行8高斯聚类的核心函数
    fprintf(fid,'%.6f',Des(1));                                         % 将聚类获得的8高斯参数写入GMM8.csv
    for j = 2 : 1 : size(Des,2)
        fprintf(fid,',%.6f',Des(j));
    end
    fclose(fid);                                                        % 关闭GMM8.csv
end
end

% 8高斯聚类核心函数
% 分别对数据集中的每张图片8高斯聚类,数据集可以是训练集或测试集
function Des = Core(count, Filename)
% DCT变换初始化
WIDTH = 8;                                                              % 窗口边长像素数
cosIU_JV = zeros([WIDTH WIDTH]);
for pk = 0 : 1 : WIDTH - 1
    for pw = 0 : 1 : WIDTH - 1
        cosIU_JV(pk + 1, pw + 1) = cos((pk + 0.5) * pi / WIDTH * pw);   % DCT非线性变换矩阵构建
    end
end
cUV = ones([WIDTH WIDTH]) * 2 / WIDTH;                                  % DCT线性变换矩阵构建
cUV(1,:) = cUV(1,:) / sqrt(2);
cUV(:,1) = cUV(:,1) / sqrt(2);
cUV(1:1) = 1 / WIDTH;
fprintf('Initialized');                                                 % 用于查看程序执行状态,可以注释掉
% 图片处理
image = imread(Filename);
% RGB YBR 变换
YBR = [0.257, 0.504, 0.098; -0.148, -0.291, 0.439; 0.439, -0.368, -0.071] * double(reshape(permute(image,[3 1 2]),3,numel(image) / 3));
YBR(1,:) = YBR(1,:) + 16;
YBR(2:3,:) = YBR(2:3,:) + 128;
YBR = reshape(YBR',size(image,1),size(image,2),3);
Px = floor((size(YBR,1) - 2) / 6);                                      % 计算X轴小窗数
Py = floor((size(YBR,2) - 2) / 6);                                      % 计算Y轴小窗数
X = zeros(18, Px * Py);                                                 % 存储DCT变换结果作为样本点
% DCT变换
for px = 0 : 1 : Px - 1
    for py = 0 : 1 : Py - 1
        block = YBR(px * 6 + 1 : px * 6 + WIDTH, py * 6 + 1 : py * 6 + WIDTH, :);
        FDCT(:,:,1) = cUV .* (cosIU_JV' * block(:,:,1) * cosIU_JV);
        FDCT(:,:,2) = cUV .* (cosIU_JV' * block(:,:,2) * cosIU_JV);
        FDCT(:,:,3) = cUV .* (cosIU_JV' * block(:,:,3) * cosIU_JV);
        % 降维,提取DCT结果的低18维(18 = 3 * 6)
        X(:, px * Py + py + 1) = reshape(permute([FDCT(1,1,:) FDCT(2,1,:) FDCT(1,2,:) FDCT(3,1,:) FDCT(2,2,:) FDCT(1,3,:)],[3 2 1]),18,1);
    end
end
fprintf('\t\tDCT Transform Finished!');                                 % 用于查看程序执行状态,可以注释掉
delSG = eye(18) * 1e-5;                                                 % 正则项,防止出现奇异矩阵
% 初始化混合高斯参数
PI = ones(1,4) * 0.125;
PI = kmeans(X, 4);
PI = PI';
Xtmp = [X zeros(18,4 - mod(size(X,2),4))];
Xtmp = reshape(Xtmp,18,size(Xtmp,2)/4,4);
MU = reshape(mean(Xtmp,2),18,4);
SG = repmat(eye(18),1,1,4) * 250000;
for k = 1:1:4
    SG(:,:,k) = cov(Xtmp(:,:,k)');
end
% 初始化其他中间变量
GM = zeros(Px * Py,4);
N = Px * Py;
L = 1;
PI_2 = (2 * pi) ^ 9;
% E-M算法
while 1
    % E - Step
    for k = 1 : 1 : 4
        GM(:,k) = PI(k) * exp(-0.5 * diag((X - repmat(MU(:,k),1,N))' / (SG(:,:,k) + delSG) * (X - repmat(MU(:,k),1,N)))) / PI_2 / sqrt(det(SG(:,:,k) + delSG));
    end
    tmpGM = sum(GM,2);
    if abs(sum(log(tmpGM)) - L) > 1e-20                                 % 似然函数增量判决(终止条件判决)
        L = sum(log(tmpGM));
    else
        break;
    end
    GM = GM ./ repmat(tmpGM,1,4);
    % M - Step
    NK = sum(GM);
    PI = NK / N;
    MU = X * GM ./ repmat(NK,18,1);
    for k = 1 : 1 : 4
        SG(:,:,k) = (X - repmat(MU(:,k),1,N)) * diag(GM(:,k)) * (X - repmat(MU(:,k),1,N))' / NK(k);
    end
end
Des = [PI reshape(MU, 1, numel(MU)) reshape(SG, 1, numel(SG))];         % 将混合高斯参数权重,期望,协方差矩阵规约为一行
fprintf('\t\tGMM-4 Finished!\t\t%4d Photos Processed!\n',count);        % 用于查看程序执行状态,可以注释掉
end
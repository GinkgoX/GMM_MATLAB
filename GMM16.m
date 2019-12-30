% 16高斯聚类函数
% 对数据集中所有图片的4高斯参数做16高斯聚类,数据集只能是训练集
function GMM16(path, type)
% 读取4高斯参数并恢复权重,期望,协方差矩阵
% 无需对4高斯分量加以区别,每个4高斯分量作为一个样本点
% 假设数据集中有N张图片,则有4N个样本点
path = [path 'train\'];
CSV = csvread(strcat(path,type,'\GMM4.csv'));
K = 4;
M = 16;
N = size(CSV,1);
LENGTH = K * N;                                                         % 样本点数4N
WIDTH = floor(sqrt(size(CSV,2) / K));                                   % 计算样本点维度
PI = reshape(CSV(:,1:K)',LENGTH,1);
MU = reshape(CSV(:,K+1:K+WIDTH*K)',WIDTH,LENGTH);
SG = reshape(CSV(:,K+WIDTH*K+1:size(CSV,2))',WIDTH,WIDTH,LENGTH);
% 初始化16混合高斯参数
H = zeros(LENGTH,M);
delSG = eye(WIDTH) * 1e-5;
cPI = ones(1,M) / M;
MUtmp = [MU zeros(18,M - mod(LENGTH,M))];
MUtmp = permute(reshape(MUtmp,18,M,size(MUtmp,2)/M),[1 3 2]);
cMU = reshape(mean(MUtmp,2),WIDTH,M);
cSG = zeros(WIDTH,WIDTH,M);
for m = 1 : 1 : M
    cSG(:,:,m) = eye(WIDTH) * 1e4;
end
L = 1;
PI_2 = (2 * pi) ^ 9;
while 1
    % E - Step
    for m = 1 : 1 : M
        for n = 1 : 1 : LENGTH
            Gauss = exp(-0.5 * diag((MU(:,n) - cMU(:,m))' / (cSG(:,:,m) + delSG) * (MU(:,n) - cMU(:,m)))) / PI_2 / sqrt(det(cSG(:,:,m) + delSG));
            H(n,m) = (Gauss * exp(-0.5 * trace((cSG(:,:,m) + delSG) \ SG(:,:,n)))) ^ PI(n) * cPI(m);
        end
    end
    tmpH = sum(H,2);
    if abs(sum(log(tmpH)) - L) > 1e-10                                  % 终止条件判决
        abs(sum(log(tmpH)) - L)
        L = sum(log(tmpH));
    else
        break;
    end
    H = H ./ repmat(tmpH,1,M);
    % M - Step
    cPI = reshape(sum(H) / LENGTH, 1, M);
    W = H .* repmat(PI,1,M) ./ repmat(PI' * H,LENGTH,1);
    cMU = MU * W;
    for m = 1 : 1 : M
        cSG(:,:,m) = reshape(reshape(SG,WIDTH * WIDTH,LENGTH) * W(:,m),WIDTH,WIDTH) + (MU - repmat(cMU(:,m),1,LENGTH)) * diag(W(:,m)) * (MU - repmat(cMU(:,m),1,LENGTH))';
    end
end
% 将16高斯聚类结果写入GMM16.csv
fid = fopen(strcat(path,type,'\GMM16.csv'),'w');
fprintf(fid,'COUNT 0,,PI,%d,MU,%d,SIGMA,%d',sum(cPI == 0),sum(sum(cMU == 0)),sum(sum(sum(cSG == 0))));
for n = 1 : 1 : M
    fprintf(fid,'\n\nGROUP,%d\nPI,%.6f',n,cPI(n));
    fprintf(fid,'\nMU');
    for i = 1 : 1 : WIDTH
        fprintf(fid,',%.6f',cMU(i,n));
    end
    for i = 1 : 1 : WIDTH
        fprintf(fid,'\n');
        if i == 1
            fprintf(fid,'SIGMA');
        end
        for j = 1 : 1 : WIDTH
            fprintf(fid,',%.6f',cSG(i,j,n));
        end
    end
end
fclose(fid);
end
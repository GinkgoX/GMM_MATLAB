function Decide4_16(path, type1, type2)
K = 4;
M = 16;
% 读测试集数据(4-GMM结果)
[testPI1, testMU1, testSG1, N1, WIDTH] = readGMM4(path, 'test', type1, K);
[testPI2, testMU2, testSG2, N2, tWIDTH] = readGMM4(path, 'test', type2, K);
if WIDTH ~= tWIDTH
    printf('Error')                                                     % 数据维度判断
    return
end
% 读训练集数据(64-GMM结果)
[trainPI1, trainMU1, trainSG1] = readGMM16(path, 'train', type1, M, WIDTH);
[trainPI2, trainMU2, trainSG2] = readGMM16(path, 'train', type2, M, WIDTH);
% 获取测试集相对某概念的概率结果
R1on1 = Core(testPI1, testMU1, testSG1, trainPI1, trainMU1, trainSG1, N1, K, M, WIDTH);
R1on2 = Core(testPI1, testMU1, testSG1, trainPI2, trainMU2, trainSG2, N1, K, M, WIDTH);
R2on1 = Core(testPI2, testMU2, testSG2, trainPI1, trainMU1, trainSG1, N2, K, M, WIDTH);
R2on2 = Core(testPI2, testMU2, testSG2, trainPI2, trainMU2, trainSG2, N2, K, M, WIDTH);
% 比较判决(分类)并将结果写入文件
R1 = R1on1 > R1on2;
R2 = R2on1 < R2on2;
type1 = lower(type1);
type2 = lower(type2);
type1(1) = upper(type1(1));
type2(1) = upper(type2(1));
fid = fopen(strcat([path 'Result of ' type1 ' vs ' type2 '.csv']),'w');
fprintf(fid,'Item,,%s\nAccuracy,,%.2f%%\nDecide,',type1, sum(R1) / size(R1,2) * 100);
for i = 1 : 1 : size(R1,2)
    if R1(i)
        fprintf(fid,',%s',type1);
    else
        fprintf(fid,',%s',type2);
    end
end
fprintf(fid,'\n\nItem,,%s\nAccuracy,,%.2f%%\nDecide,',type2, sum(R2) / size(R2,2) * 100);
for i = 1 : 1 : size(R2,2)
    if R2(i)
        fprintf(fid,',%s',type2);
    else
        fprintf(fid,',%s',type1);
    end
end
fprintf(fid,'\n\nAccuracy,,%.2f%%', (sum(R1) + sum(R2)) / (size(R1,2) + size(R2,2)) * 100);
fclose(fid);
end

% 读取GMM4.csv内容,恢复4高斯的权重,期望,协方差矩阵
function [PI, MU, SG, N, WIDTH] = readGMM4(path, set, type, K)
CSV = csvread(strcat(path, set, '\', type, '\GMM4.csv'));
N = size(CSV,1);
WIDTH = floor(sqrt(size(CSV,2) / K));
PI = CSV(:,1:K);
MU = reshape(CSV(:,K+1:K+WIDTH*K)',WIDTH,K,N);
SG = reshape(CSV(:,K+WIDTH*K+1:size(CSV,2))',WIDTH,WIDTH,K,N);
end

% 读取GMM64.csv内容,恢复16高斯的权重,期望,协方差矩阵
function [PI, MU, SG] = readGMM16(path, set, type, M, WIDTH)
fid = fopen(strcat(path, set, '\', type, '\GMM16.csv'),'r');
CSV = textscan(fid, '%s');
CSV = CSV{1};
PI = zeros(1, M);
MU = zeros(WIDTH, M);
SG = zeros(WIDTH, WIDTH, M);
for n = 1 : 1 : M
    srcGroup = CSV(21 * n - 18 : 21 * n + 2);
    text = srcGroup(2);
    text = text{1};
    PI(n) = str2double(text(4 : length(text)));
    text = srcGroup(3);
    text = text{1};
    MU(:, n) = str2double(regexp(text(4 : length(text)), ',', 'split'))';
    for w = 1 : 1 : WIDTH
        text = srcGroup(3 + w);
        text = text{1};
        SG(w, :, n) = str2double(regexp(text(7 : length(text)), ',', 'split'));
    end
end
fclose(fid);
end

% 分类判决核心函数
function Core = Core(PI, MU, SG, cPI, cMU, cSG, N, K, M, WIDTH)
delSG = eye(WIDTH) * 1e-5;
H = zeros(N,K,M);
% 计算测试集相对某概念的概率
for n = 1 : 1 : N
    for k = 1 : 1 : K
        for m = 1 : 1 : M
            Gauss = exp(-0.5 * diag((MU(:,k,n) - cMU(:,m))' / (cSG(:,:,m) + delSG) * (MU(:,k,n) - cMU(:,m)))) / 2 / pi / sqrt(det(cSG(:,:,m) + delSG));
            H(n,k,m) = (Gauss * exp(-0.5 * trace((cSG(:,:,m) + delSG) \ SG(:,:,k,n)))) ^ PI(n,k) * cPI(m);
        end
    end
end
% 反馈概率值
Core = sum(sum(H,3) .* PI, 2)';
end
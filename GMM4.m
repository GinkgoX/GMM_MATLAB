function GMM8(path, set, type)
% ��ȡ���ݼ�ͼƬ�б�
fileExt = '*.jpeg';
files = dir(fullfile(path, set, '\', type, '\', fileExt));
% �������ݼ��е�ͼƬ8��˹��������
for i = 1 : 1 : size(files,1)
    if i == 1
        fid = fopen(strcat(path,set,'\',type,'\GMM4.csv'),'w');         % ��һ��ͼƬ�������½��򸲸���ʽд��GMM8.csv
    else
        fid = fopen(strcat(path,set,'\',type,'\GMM4.csv'),'a');         % ����ͼƬ������׷����ʽд��GMM8.csv
        fprintf(fid,'\n');
    end
    fprintf('%s\n',strcat(path,type,files(i).name));                    % ���ڲ鿴����ִ��״̬(��ʾ�ļ���),����ע�͵�
    Des = Core(i, strcat(path,set,'\',type,'\',files(i).name));         % ִ��8��˹����ĺ��ĺ���
    fprintf(fid,'%.6f',Des(1));                                         % �������õ�8��˹����д��GMM8.csv
    for j = 2 : 1 : size(Des,2)
        fprintf(fid,',%.6f',Des(j));
    end
    fclose(fid);                                                        % �ر�GMM8.csv
end
end

% 8��˹������ĺ���
% �ֱ�����ݼ��е�ÿ��ͼƬ8��˹����,���ݼ�������ѵ��������Լ�
function Des = Core(count, Filename)
% DCT�任��ʼ��
WIDTH = 8;                                                              % ���ڱ߳�������
cosIU_JV = zeros([WIDTH WIDTH]);
for pk = 0 : 1 : WIDTH - 1
    for pw = 0 : 1 : WIDTH - 1
        cosIU_JV(pk + 1, pw + 1) = cos((pk + 0.5) * pi / WIDTH * pw);   % DCT�����Ա任���󹹽�
    end
end
cUV = ones([WIDTH WIDTH]) * 2 / WIDTH;                                  % DCT���Ա任���󹹽�
cUV(1,:) = cUV(1,:) / sqrt(2);
cUV(:,1) = cUV(:,1) / sqrt(2);
cUV(1:1) = 1 / WIDTH;
fprintf('Initialized');                                                 % ���ڲ鿴����ִ��״̬,����ע�͵�
% ͼƬ����
image = imread(Filename);
% RGB YBR �任
YBR = [0.257, 0.504, 0.098; -0.148, -0.291, 0.439; 0.439, -0.368, -0.071] * double(reshape(permute(image,[3 1 2]),3,numel(image) / 3));
YBR(1,:) = YBR(1,:) + 16;
YBR(2:3,:) = YBR(2:3,:) + 128;
YBR = reshape(YBR',size(image,1),size(image,2),3);
Px = floor((size(YBR,1) - 2) / 6);                                      % ����X��С����
Py = floor((size(YBR,2) - 2) / 6);                                      % ����Y��С����
X = zeros(18, Px * Py);                                                 % �洢DCT�任�����Ϊ������
% DCT�任
for px = 0 : 1 : Px - 1
    for py = 0 : 1 : Py - 1
        block = YBR(px * 6 + 1 : px * 6 + WIDTH, py * 6 + 1 : py * 6 + WIDTH, :);
        FDCT(:,:,1) = cUV .* (cosIU_JV' * block(:,:,1) * cosIU_JV);
        FDCT(:,:,2) = cUV .* (cosIU_JV' * block(:,:,2) * cosIU_JV);
        FDCT(:,:,3) = cUV .* (cosIU_JV' * block(:,:,3) * cosIU_JV);
        % ��ά,��ȡDCT����ĵ�18ά(18 = 3 * 6)
        X(:, px * Py + py + 1) = reshape(permute([FDCT(1,1,:) FDCT(2,1,:) FDCT(1,2,:) FDCT(3,1,:) FDCT(2,2,:) FDCT(1,3,:)],[3 2 1]),18,1);
    end
end
fprintf('\t\tDCT Transform Finished!');                                 % ���ڲ鿴����ִ��״̬,����ע�͵�
delSG = eye(18) * 1e-5;                                                 % ������,��ֹ�����������
% ��ʼ����ϸ�˹����
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
% ��ʼ�������м����
GM = zeros(Px * Py,4);
N = Px * Py;
L = 1;
PI_2 = (2 * pi) ^ 9;
% E-M�㷨
while 1
    % E - Step
    for k = 1 : 1 : 4
        GM(:,k) = PI(k) * exp(-0.5 * diag((X - repmat(MU(:,k),1,N))' / (SG(:,:,k) + delSG) * (X - repmat(MU(:,k),1,N)))) / PI_2 / sqrt(det(SG(:,:,k) + delSG));
    end
    tmpGM = sum(GM,2);
    if abs(sum(log(tmpGM)) - L) > 1e-20                                 % ��Ȼ���������о�(��ֹ�����о�)
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
Des = [PI reshape(MU, 1, numel(MU)) reshape(SG, 1, numel(SG))];         % ����ϸ�˹����Ȩ��,����,Э��������ԼΪһ��
fprintf('\t\tGMM-4 Finished!\t\t%4d Photos Processed!\n',count);        % ���ڲ鿴����ִ��״̬,����ע�͵�
end
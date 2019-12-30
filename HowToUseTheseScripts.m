clc;
clear;

% 这是一个使用示例,仅用于说明函数调用的方法.
% 由于运行时间很长,实际是在控制台分步调用各模块的.

path = '\your\path\to\GMM\dataset\';               % 指定数据即路径
GMM4(path, 'train', 'class1');           % class1训练集4高斯聚类
GMM4(path, 'train', 'class2');           % class2训练集4高斯聚类
GMM4(path, 'test', 'class1');            % class1测试集4高斯聚类
GMM4(path, 'test', 'class2');            % class2测试集4高斯聚类
GMM16(path,'class1');                    % class1训练集16高斯聚类
GMM16(path,'class2');                    % class2训练集16高斯聚类
Decide4_16(path, 'class1', 'class2');        % 对class1和class2测试集分类判决
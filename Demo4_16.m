clc;
clear;

% 这是一个使用示例,仅用于说明函数调用的方法.
% 由于运行时间很长,实际是在控制台分步调用各模块的.

path = 'C:\Users\ginkgoX\Documents\MATLAB\GMM\dataset\';               % 指定数据即路径
GMM4(path, 'train', 'human');           % 人类训练集4高斯聚类
GMM4(path, 'train', 'tiger');           % 老虎训练集4高斯聚类
GMM4(path, 'test', 'human');            % 人类测试集4高斯聚类
GMM4(path, 'test', 'tiger');            % 老虎测试集4高斯聚类
GMM16(path,'human');                    % 人类训练集16高斯聚类
GMM16(path,'tiger');                    % 老虎训练集16高斯聚类
Decide4_16(path, 'human', 'tiger');         % 对人类和老虎测试集分类判决
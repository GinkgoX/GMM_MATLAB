clc;
clear;

% ����һ��ʹ��ʾ��,������˵���������õķ���.
% ��������ʱ��ܳ�,ʵ�����ڿ���̨�ֲ����ø�ģ���.

path = 'C:\Users\ginkgoX\Documents\MATLAB\GMM\dataset\';               % ָ�����ݼ�·��
GMM4(path, 'train', 'human');           % ����ѵ����4��˹����
GMM4(path, 'train', 'tiger');           % �ϻ�ѵ����4��˹����
GMM4(path, 'test', 'human');            % ������Լ�4��˹����
GMM4(path, 'test', 'tiger');            % �ϻ����Լ�4��˹����
GMM16(path,'human');                    % ����ѵ����16��˹����
GMM16(path,'tiger');                    % �ϻ�ѵ����16��˹����
Decide4_16(path, 'human', 'tiger');         % ��������ϻ����Լ������о�
clc;
clear;

% ����һ��ʹ��ʾ��,������˵���������õķ���.
% ��������ʱ��ܳ�,ʵ�����ڿ���̨�ֲ����ø�ģ���.

path = '\your\path\to\GMM\dataset\';               % ָ�����ݼ�·��
GMM4(path, 'train', 'class1');           % class1ѵ����4��˹����
GMM4(path, 'train', 'class2');           % class2ѵ����4��˹����
GMM4(path, 'test', 'class1');            % class1���Լ�4��˹����
GMM4(path, 'test', 'class2');            % class2���Լ�4��˹����
GMM16(path,'class1');                    % class1ѵ����16��˹����
GMM16(path,'class2');                    % class2ѵ����16��˹����
Decide4_16(path, 'class1', 'class2');        % ��class1��class2���Լ������о�
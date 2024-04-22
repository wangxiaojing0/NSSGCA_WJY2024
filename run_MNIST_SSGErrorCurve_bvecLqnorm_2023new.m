%--------------------------------------------------------------------------
% This is the main function to run the SCLRSmC algorithm for the image
% clustering problem on the UCSD sample dataset with 4 classes.
%
% Version
% --------
% Companion Code Version: 1.0
%
%
% Citation
% ---------
% Any part of this code used in your work should be cited as follows:
%
% T. Wu and W. U. Bajwa, "A low tensor-rank representation approach for clustering of imaging data,"
% IEEE Signal Processing Letters, vol. 25, no. 8, pp. 1196-1200, 2018, Companion Code, ver. 1.0.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% MIT License
%
% Copyright <2022> <Tong Wu and Waheed U. Bajwa>
%
% Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
% associated documentation files (the "Software"), to deal in the Software without restriction, including 
% without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
% copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
% OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
% LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
% IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


clearvars;
close all;
clc

load('MNISTclass10per100.mat');

data=X_MNISTclass10per100;
clustern=max(unique(gnd));
gt=gnd;

%%
vecdata = zeros(size(data,1)*size(data,3), size(data,2));
for i = 1:size(data,2)
    temp = data(:,i,:);
    vecdata(:,i) = temp(:);
end

vecdata = vecdata./repmat(sqrt(sum(vecdata.^2)),[size(data,1)*size(data,3) 1]);
L = vecdata'*vecdata;
sigma = mean(mean(1 - L));
M = ones(size(L)) - exp(-(ones(size(L))-L)/sigma);
M=M-diag(diag(M));
%% MNIST best

lambda1 = 0.2;
lambda2 = 0.04;
lambda3=0.3;
mu = 0.1;
mu2=0.1;
rho = 1.9;


[Z,iter,Error_ZC,Error_ZQ,Error_ZT,Error_SSG,~] = SCLRSmCWJYSSGnew_bvec_Lqnorm_ErrorCurve2023MNIST(data, M, lambda1, lambda2, mu, rho, mu2, lambda3);
affi = sqrt(sum(Z.^2,3));
affi = affi + affi';


%% measure methods
ACC_total = zeros(1,20);
NMI_total = zeros(1,20);
Purity_total = zeros(1,20);
F_total = zeros(1,20);
P_total = zeros(1,20);
R_total = zeros(1,20);
RI_total = zeros(1,20);


for i_ave = 1:20
    grps = SpectralClustering(affi,clustern);
    result =  ClusteringMeasure(gt, grps);
    ACC_total(i_ave) = result(1,1);
    NMI_total(i_ave) = result(1,2);
    Purity_total(i_ave) = result(1,3);
    F_total(i_ave) = result(1,4);
    P_total(i_ave) = result(1,5);
    R_total(i_ave) = result(1,6);
    RI_total(i_ave) = result(1,7);
%     AR_total(i_ave) = AR;
%     Recall_total(i_ave) = r;
end
ACC_mean = mean(ACC_total); ACC_std = std(ACC_total);
NMI_mean = mean(NMI_total); NMI_std = std(NMI_total);
Purity_mean = mean(Purity_total); Purity_std = std(Purity_total);
F_mean = mean(F_total); F_std = std(F_total);
P_mean = mean(P_total); P_std = std(P_total);
R_mean = mean(R_total); R_std = std(R_total);
RI_mean = mean(RI_total); RI_std = std(RI_total);
% AVG_mean = mean(AVG_total); AVG_std = std(AVG_total);
% Precision_mean = mean(Precision_total); Precision_std = std(Precision_total);
% RI_mean = mean(RI_total); RI_std = std(RI_total);
% AR_mean = mean(AR_total); AR_std = std(AR_total);
% Recall_mean = mean(Recall_total); Recall_std = std(Recall_total);

fprintf('ACC: %f(%f), NMI: %f(%f),  Purity: %f(%f), F-Score: %f(%f), P: %f(%f), R: %f(%f), RI: %f(%f)\n',...
    ACC_mean,ACC_std,NMI_mean,NMI_std,Purity_mean, Purity_std, F_mean,F_std,P_mean,P_std,R_mean,R_std,RI_mean,RI_std);

t=1:1:iter-1;

figure(2)
plot(Error_ZC,'--d','color',[34 139 34]/255,'LineWidth',1,'MarkerSize',4); % 缁胯壊
hold on
plot(Error_ZQ,'--o','color',[205 92 92]/255,'LineWidth',1,'MarkerSize',4); % 鐜孩
hold on
plot(Error_ZT,'--^','color',[255 174 0]/255,'LineWidth',1,'MarkerSize',4); % orange 姗橀粍鑹?
hold on
plot(Error_SSG,'--+','color',[0.3 0.5 1],'LineWidth',1,'MarkerSize',4); % orange 姗橀粍鑹?
xlabel('Number of iterations')
ylabel('Error')
legend('\fontname{Times New Roman}||Z-C||_{\infty}','\fontname{Times New Roman}||Z-Q||_{\infty}','\fontname{Times New Roman}||Z-T||_{\infty}','\fontname{Times New Roman}||T^{(k)}K^{T}-J^{(k)}||_{\infty}')


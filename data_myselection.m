clc;clear all;close all;

load data
%%
features = mydata(:,1:10);
label    = mydata(:,11);
Fs = 16;
Ft = 20;
%% A. outlier
idx_age  = find(mydata(:,2)<10); % age outliers
idx_BMI = find(mydata(:,9)>60); % BMI outliers

mydata(idx_age,:) = [];

%% B. Split 
train_data = mydata(floor(1:4/5*size(mydata,1)),:);
test_data  = mydata(floor(4/5*size(mydata,1))+1:end,:);

f_train = train_data(:,1:end-1);
l_train = train_data(:,end);

f_test = test_data(:,1:end-1);
l_test = test_data(:,end);

feat = [f_train;f_test];
lab  = [l_train;l_test];


%% C. Normalization
% z-score
f_trainN = zscore(f_train);
mu = mean(f_trainN);
STD = std(f_trainN);

for i=1:size(f_test,2)
    f_testN(:,i)=(f_test(:,i)-mu(i))/STD(i);
end
feat_N = [f_trainN;f_testN];

% minmax
f_trainNM = (mapminmax(f_train',0,1))';
f_testNM = (mapminmax(f_test',0,1))';
feat_NM = [f_trainNM;f_testNM];

%% save cleaned, normalized Data
mydata    = [feat,lab];   % without normalization
mydata_N  = [feat_N,lab]; % normal with z-score
mydata_NM = [feat_NM,lab];% normal with minmax
save mydata mydata mydata_N mydata_NM 
%% D. Feature Selection Visualization
% 1.
figure()
[idx,z] = rankfeatures(f_train',l_train,'criterion','ttest');
subplot(1,2,1)
bar(z)
title('feature ranking',...
      'fontsize',Ft,...
      'FontName','Times New Roman');
xlabel('index of features',...
       'FontWeight','bold',...
       'fontsize',Fs,...
       'FontName','Times New Roman');
ylabel('priority','FontSize',Fs,...
       'FontWeight','bold',...
       'FontName','Times New Roman');
ylim([0 12])
% 2. 
mdl = fscnca(f_train,l_train,'solver','sgd');
subplot(1,2,2)
plot(mdl.FeatureWeights,'r-o')
grid on
title('feature ranking',...
      'fontsize',Ft,...
      'FontName','Times New Roman');
xlabel('index of features',...
       'FontWeight','bold',...
       'fontsize',Fs,...
       'FontName','Times New Roman');
ylabel('weight','FontSize',Fs,...
       'FontWeight','bold',...
       'FontName','Times New Roman');
xlim([1 10])
ylim([0 2.5])

% 3. PCA
[coeff,scoreTrain,~,~,explained,mu] = pca(f_trainNM);
explained
idx = find(cumsum(explained)>95,1)
scoreTrain95 = scoreTrain(:,1:idx);

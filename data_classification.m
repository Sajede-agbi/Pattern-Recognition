clc;clear all;close all;

load mydata
%%
features = mydata(:,1:10);
label    = mydata(:,11);
Fs = 16;
Ft = 20;
%% Splitting
train_data = mydata(floor(1:4/5*size(mydata,1)),:);
test_data  = mydata(floor(4/5*size(mydata,1))+1:end,:);

f_train = train_data(:,1:end-1);
l_train = train_data(:,end);

f_test = test_data(:,1:end-1);
l_test = test_data(:,end);

feat = [f_train;f_test];
lab  = [l_train;l_test];
train_data = [f_train,l_train];

%% PCA

[coeff,scoreTrain,~,~,explained,mu] = pca(f_train);
explained
idx = find(cumsum(explained)>95,1)
scoreTrain95 = scoreTrain(:,1:idx);
%% training model & validation
% K-fold cross validation

K = 10;
n_run = 3;
accuracy = zeros(K,n_run);
% 10_fold
for i_run=1:n_run
    indices = crossvalind('Kfold',l_train,K);
    
    for i_fold = 1:K
        Val = indices==i_fold;
        train = ~Val;
        featureTrain = f_train(train,:);
        featureVal = f_train(Val,:);
        
        %% Classification
        % 1. Naive Baysian
            NBmodel  = fitcnb(f_train,l_train);
        % 2. KNN
            KNNmodel = fitcknn(f_train,l_train,'NumNeighbors',5,'distance','minkowski');
        % 3. SVM
            SVMmodel = fitcsvm(f_train,l_train,'KernelFunction','linear');   % gaussian
        % 4. LDA
            LDmodel  = fitcdiscr(f_train,l_train,'DiscrimType','quadratic');    % quadratic 
        % 5. DT
            DTmodel  = fitctree(f_train,l_train);
         
% PCA is on
%         % 1. Naive Baysian
%             NBmodel  = fitcnb(scoreTrain95,l_train);
%         % 2. KNN
%             KNNmodel = fitcknn(scoreTrain95,l_train,'NumNeighbors',5,'distance','minkowski'); %parameters tuned
%         % 3. SVM
%             SVMmodel = fitcsvm(scoreTrain95,l_train,'KernelFunction','linear');   % gaussian>>weak
%         % 4. LDA
%             LDmodel  = fitcdiscr(scoreTrain95,l_train,'DiscrimType','quadratic'); % linear >> weak 
%         % 5. DT
%             DTmodel  = fitctree(scoreTrain95,l_train);

    end
end

%% predicting 

[bp_test,~,~] = predict(NBmodel,f_test);
[kp_test,~,~] = predict(KNNmodel,f_test);   
[sp_test,~,~] = predict(SVMmodel,f_test);
[lp_test,~,~] = predict(LDmodel,f_test);
[tp_test,~,~] = predict(DTmodel,f_test);

% PCA on

%scoreTest95 = (f_test-mu)*coeff(:,1:idx);
% [bp_test,~,~] = predict(NBmodel,scoreTest95);
% [kp_test,~,~] = predict(KNNmodel,scoreTest95);   
% [sp_test,~,~] = predict(SVMmodel,scoreTest95);
%[lp_test,~,~] = predict(LDmodel,scoreTest95);
% [tp_test,~,~] = predict(DTmodel,scoreTest95);

%% 1. NB 
% Confusion matrix of Naive Baysian
figure()
plotconfusion(l_test',bp_test')
title('Confusion matrix of NB classification',...
      'fontsize',Ft,...
      'FontName','Times New Roman');
[c order] = confusionmat(l_test,bp_test);
TN=c(1,1);
FP=c(2,1);
FN=c(1,2);
TP=c(2,2);
acc         = ((TP+TN)/length(l_test))*100  %accuracy
perc        = (TP / (TP + FP))*100          %precision  
spec        = (TN / (TN+FP))*100            %specialty
f1measureknn=(2*TP)/(2*TP+FN+FP)*100        %F1
recall      = (TP / (TP+FN))*100            %recall score

% ROC of NB
[x,y, ~,aucNB] = perfcurve(l_test',bp_test',1);
AUCNB_b=aucNB*100;
figure
plotroc(l_test',bp_test');
legend('random classifier',['NB AUC=',num2str(AUCNB_b)])

title('ROC of NB classification',...
      'fontsize',Ft,...
      'FontName','Times New Roman');
xlabel('False Positive Rate','FontSize',Fs,...
       'FontWeight','bold',...
       'FontName','Times New Roman');
ylabel('True Positive Rate','FontSize',Fs,...
       'FontWeight','bold',...
       'FontName','Times New Roman');
 

%%  2.KNN
% Confusion matrix of KNN
figure()
plotconfusion(l_test',kp_test')
title('Confusion matrix of KNN classification',...
      'fontsize',Ft,...
      'FontName','Times New Roman');
[c order] = confusionmat(l_test,kp_test);
TN=c(1,1);
FP=c(2,1);
FN=c(1,2);
TP=c(2,2);
acc         = ((TP+TN)/length(l_test))*100  %accuracy
perc        = (TP / (TP + FP))*100          %precision  
spec        = (TN / (TN+FP))*100            %specialty
f1measureknn=(2*TP)/(2*TP+FN+FP)*100        %F1
recall      = (TP / (TP+FN))*100            %recall score
%  ROC of KNN
[x,y, ~,aucNB] = perfcurve(l_test',kp_test',1);
AUCNB_k=aucNB*100;
figure
plotroc(l_test',kp_test');
legend('random classifier',['KNN AUC=',num2str(AUCNB_k)])

title('ROC of KNN classification',...
      'fontsize',Ft,...
      'FontName','Times New Roman');
xlabel('False Positive Rate','FontSize',Fs,...
       'FontWeight','bold',...
       'FontName','Times New Roman');
ylabel('True Positive Rate','FontSize',Fs,...
       'FontWeight','bold',...
       'FontName','Times New Roman');

%% 3.SVM   
% Confusion matrix of SVM
figure()
plotconfusion(l_test',sp_test')
title('Confusion matrix of SVM classification',...
      'fontsize',Ft,...
      'FontName','Times New Roman');
[c order] = confusionmat(l_test,sp_test);
TN=c(1,1);
FP=c(2,1);
FN=c(1,2);
TP=c(2,2);
acc         = ((TP+TN)/length(l_test))*100  %accuracy
perc        = (TP / (TP + FP))*100          %precision  
spec        = (TN / (TN+FP))*100            %specialty
f1measureknn=(2*TP)/(2*TP+FN+FP)*100        %F1
recall      = (TP / (TP+FN))*100            %recall score

% ROC of SVM
[x,y, ~,aucNB] = perfcurve(l_test',sp_test',1);
AUCNB_s=aucNB*100;
figure
plotroc(l_test',sp_test');
legend('random classifier',['SVM AUC=',num2str(AUCNB_s)])

title('ROC of SVM classification',...
      'fontsize',Ft,...
      'FontName','Times New Roman');
xlabel('False Positive Rate','FontSize',Fs,...
       'FontWeight','bold',...
       'FontName','Times New Roman');
ylabel('True Positive Rate','FontSize',Fs,...
       'FontWeight','bold',...
       'FontName','Times New Roman');

%% 4. LDA
%  Confusion matrix of LDA
figure()
plotconfusion(l_test',lp_test')
title('Confusion matrix of LDA classification',...
      'fontsize',Ft,...
      'FontName','Times New Roman');
[c order] = confusionmat(l_test,lp_test);
TN=c(1,1);
FP=c(2,1);
FN=c(1,2);
TP=c(2,2);
acc         = ((TP+TN)/length(l_test))*100  %accuracy
perc        = (TP / (TP + FP))*100          %precision  
spec        = (TN / (TN+FP))*100            %specialty
f1measureknn=(2*TP)/(2*TP+FN+FP)*100        %F1
recall      = (TP / (TP+FN))*100            %recall score
% ROC of LDA
[x,y, ~,aucNB] = perfcurve(l_test',lp_test',1);
AUCNB_l=aucNB*100;
figure
plotroc(l_test',lp_test');
legend('random classifier',['LDA AUC=',num2str(AUCNB_l)])

title('ROC of LDA classification',...
      'fontsize',Ft,...
      'FontName','Times New Roman');
xlabel('False Positive Rate','FontSize',Fs,...
       'FontWeight','bold',...
       'FontName','Times New Roman');
ylabel('True Positive Rate','FontSize',Fs,...
       'FontWeight','bold',...
       'FontName','Times New Roman');
%% 5.Decsion Tree
%  Confusion matrix of DT
figure()
plotconfusion(l_test',tp_test')
title('Confusion matrix of DT classification',...
      'fontsize',Ft,...
      'FontName','Times New Roman');
[c order] = confusionmat(l_test,tp_test);
TN=c(1,1);
FP=c(2,1);
FN=c(1,2);
TP=c(2,2);
acc         = ((TP+TN)/length(l_test))*100  %accuracy
perc        = (TP / (TP + FP))*100          %precision  
spec        = (TN / (TN+FP))*100            %specialty
f1measureknn=(2*TP)/(2*TP+FN+FP)*100        %F1
recall      = (TP / (TP+FN))*100            %recall score
% ROC of DT
[x,y, ~,aucNB] = perfcurve(l_test',tp_test',1);
AUCNB_t=aucNB*100;
figure
plotroc(l_test',tp_test');
legend('random classifier',['LDA AUC=',num2str(AUCNB_t)])

title('ROC of DT classification',...
      'fontsize',Ft,...
      'FontName','Times New Roman');
xlabel('False Positive Rate','FontSize',Fs,...
       'FontWeight','bold',...
       'FontName','Times New Roman');
ylabel('True Positive Rate','FontSize',Fs,...
       'FontWeight','bold',...
       'FontName','Times New Roman');
  
%% ROCs plot
[x1,y1, ~,~] = perfcurve(l_test',bp_test',1);
[x2,y2, ~,~] = perfcurve(l_test',kp_test',1);
[x3,y3, ~,~] = perfcurve(l_test',sp_test',1);
[x4,y4, ~,~] = perfcurve(l_test',lp_test',1);
[x5,y5, ~,~] = perfcurve(l_test',tp_test',1);
x = [0 1];
y = x;
figure()
plot(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,'LineWidth',3)
hold on
plot(x,y,'LineWidth',3)
title('ROC of different classifiers (PCA off)',...
      'fontsize',Ft,...
      'FontName','Times New Roman');
xlabel('False Positive Rate','FontSize',Fs,...
       'FontWeight','bold',...
       'FontName','Times New Roman');
ylabel('True Positive Rate','FontSize',Fs,...
       'FontWeight','bold',...
       'FontName','Times New Roman');
legend(['NB AUC=',num2str(AUCNB_b)],...
       ['KNN AUC=',num2str(AUCNB_k)],...
       ['SVM AUC=',num2str(AUCNB_s)],...
       ['LDA AUC=',num2str(AUCNB_l)],...
       ['DT AUC=',num2str(AUCNB_t)],...
        'random classifier')
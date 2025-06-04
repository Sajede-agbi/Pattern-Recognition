clc
clear
close all

%% data loading
data_original = readtable('healthcare-dataset-stroke-data.csv');
%% A
data = data_original;
data = table2struct (data);
%% B : char 2 double
% Male = 0 & Female = 1
i = 0;
for i=1:length(data)
    if ismember(data(i).gender,'Female')
        data(i).gender = 1;
    else
        data(i).gender = 0;
    end
end

i = 0;
% ever_married
% No = 0 &  Yes= 1
for i=1:length(data)
    if ismember(data(i).ever_married,'Yes')
        data(i).ever_married = 1;
    else
        data(i).ever_married = 0;
    end
end

i = 0;
% work_type
% children      = 0 
% Never_worked  = 1
% Govt_jov      = 2
% Private       = 3
% Self-employed = 4
for i=1:length(data)
    if ismember(data(i).work_type,'children')
        data(i).work_type = 0;
    elseif ismember(data(i).work_type,'Never_worked')
        data(i).work_type = 1;
    elseif ismember(data(i).work_type,'Govt_job')
        data(i).work_type = 2;
    elseif ismember(data(i).work_type,'Private')
        data(i).work_type = 3;
    else 
        data(i).work_type = 4;
    end
end

i = 0;
% Residence_type
% Urban = 0 & Rural = 1
for i=1:length(data)
    if ismember(data(i).Residence_type,'Rural')
        data(i).Residence_type = 1;
    else
        data(i).Residence_type = 0;
    end
end

i = 0;
% smoking_status
% Unknown         = 0
% never smoked    = 1 
% formerly smoked = 2
% smokes          = 3
for i=1:length(data)
    if ismember(data(i).smoking_status,'Unknown')
        data(i).smoking_status = 0;
    elseif ismember(data(i).smoking_status,'smokes')
        data(i).smoking_status = 3 ;
    elseif ismember(data(i).smoking_status,'never smoked')
        data(i).smoking_status = 1 ;
    elseif ismember(data(i).smoking_status,'formerly smoked')
        data(i).smoking_status = 2 ;
    end
end

bmi=zeros(1,length(data));
for i=1:length(data)
    data(i).bmi=str2double(data(i).bmi);
end
%% C
data = cell2mat(struct2cell(data));
data = data';
%% %%%Pre-processing%%%%
%% A
data(:,1) = [];              % remove ID
label_original = data(:,11);
features_original = data(:,1:10);
%% B : devide healthy and stroke
features_s = features_original(1:249,:);                         % stroke
features_h = features_original(250:length(features_original),:); % healthy

%% C : removing Nan & unknowns from health data
bmi_idx_h = find(isnan(features_h(:,9)));        % NaN MBI
smoke_idx_h = find(features_h(:,10)==0);         % Unknown smoke_state
del_idx = sort([bmi_idx_h',smoke_idx_h']);

% search and remove duplicated index with both NaN & Unknown
k=0;
for i=2:length(del_idx)
    if del_idx(i-1)==del_idx(i)
       k=k+1;
        id(k)=i;
    end
end  
del_idx(id)=[];            % index of healthy data with either NaN BMI or Unknown smoking statuse

features_h(del_idx,:)=[];  % Clean Healthy Data
%% D : mean of bmi for NaN & mod for smoke statues in stroke data 

bmi_idx_s = find(isnan(features_s(:,9)));          % NaN MBI
not_NaN   = find(~isnan(features_s(:,9)));
mean_bmi  = mean( features_s(not_NaN,9));          % mean of bmi culomn
features_s(bmi_idx_s,9)= mean_bmi;                 % Clean Stroke Data

smoke_idx_s = find(features_s(:,10)==0); 
mod_smoke = mode(features_s(:,10));
features_s(smoke_idx_s,10) = mod_smoke;
% Duplicated 
s = 0;
for j=1:length(smoke_idx_s)
    for i=1:length(bmi_idx_s)
        if bmi_idx_s(i)==smoke_idx_s(j)
            s=s+1;
            remove_s(s)=smoke_idx_s(j);   % duplicated NaN smoke&BMI in stroke data
            break
        end
    end
end

%% E
%c=randperm(length(features_h),length(features_s));  % select random healthy data equal to size of stroke data
%please load data.mat to extrect c
load data
dec_features_h = features_h(c,:);                                                   % decreased healthy data 
features       = [features_s',dec_features_h']';                                    % new features
label          = [ones(length(features_s),1)',zeros(length(dec_features_h),1)']';   % new labels
mydata         = [features,label];

%% F : shuffle & save
% mydata_before_shuffling = mydata;
% mydata = mydata(randperm(size(mydata, 1)), :);
% save data data mydata_before_shuffling mydata c
% save mydata mydata
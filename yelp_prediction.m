clear all; clc; close all;



D = load('trainData.txt');
trX = sparse(D(:,2), D(:,1), D(:,3));
trLb = load('trainLabels.txt');

rt_list=[];
rv_list=[];
non_w=[];
dif_v=[];
l=[];

%5.4.1
% D = load('valData.txt');
% vX = sparse(D(:,2), D(:,1), D(:,3));
% vLb = load('valLabels.txt');
% 
% 
% lambda=trX*(trLb-sum(trLb)/10000);
% lambda=((lambda'*lambda)^0.5)*2;


% iter=10;
% for i=1:iter
%     
%     [w_train,b_train]=CD_Lasso(trLb,trX,lambda);
% 
%     y_t=trX'*w_train+b_train;
%     y_v=vX'*w_train+b_train;
% 
%     rmse_t=(mean((y_t-trLb).^2))^0.5;
%     rmse_v=(mean((y_v-vLb).^2))^0.5;
%     non_w=[non_w,size(find(w_train),1)];
%     l=[l,lambda];
%     
%     rt_list=[rt_list,rmse_t];
%     rv_list=[rv_list,rmse_v];
%     if(i<2)
%         dif_v=[dif_v,0];
%     else
%         dif_v=[dif_v,rv_list(i)-rv_list(i-1)];
%     end
%     
%     disp(['lambda: ',num2str(lambda)]);
%     disp(['round: ',num2str(i),' finished']);
%     lambda=lambda/2;
% end
% 
% fig1=figure(1);
% plot([1:iter],rv_list,'-ro',[1:iter],rt_list,'-xb');
% axis([0,iter,0,1.5]);
% set(gca,'xticklabel', [0,l]);
% legend('valid errors','train errors');
% for i=1:iter
% %   text(i,rt_list(i),[num2str(rt_list(i))]);
%     text(i,rv_list(i),[num2str(rv_list(i))]);
% end
% saveas(fig1,'result_RMSE.png');
% 
% fig2=figure(2);
% plot([1:iter],non_w,'-ro');
% axis([0,iter,0,2500]);
% set(gca,'xticklabel', [0,l]);
% legend('nonzeros');
% for i=1:iter
%     if(non_w(i)==0)
%         continue;
%     else
%         text(i,non_w(i),[num2str(non_w(i))]);
%     end
% end
% saveas(fig2,'result_nonzeros_w.png');


%5.4.2
lambda=1.66324;
[w_train,b_train]=CD_Lasso(trLb,trX,lambda);
largest_ten=sort(w_train,'descend');
largest_ten=largest_ten(1:10);
least_ten=sort(w_train,'ascend');
least_ten=least_ten(1:10);

index_largest=[];
index_least=[]
for i=1:10
    j=find(w_train==largest_ten(i));
    k=find(w_train==largest_ten(i));
    index_largest=[index_largest;j];
    index_least=[index_least;k];
end

%5.4.3
D = load('testData.txt');
testX = sparse(D(:,2), D(:,1), D(:,3));
Prediction=testX'*w_train+b_train;
instanceID=[1:25000]';

Prediction=max(Prediction,1);
Prediction=min(Prediction,5);

result=table(instanceID,Prediction);
writetable(result,'predTestLabels.csv');
type 'predTestLabels.csv';







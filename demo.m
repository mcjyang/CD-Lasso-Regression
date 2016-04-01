clc; clear all; close all;

mu=0;
simga=10;
d=75;
n=200;
pd=makedist('Normal','mu',mu,'sigma',simga)
X=random(pd,d,n);
w=zeros(d,1);
w(find(~w,10))=10;
noise=random(pd,n,1);

y=X'*w+noise;

lambda=X*(y-sum(y)/n);
lambda=((lambda'*lambda)^0.5)*2;

rw=w;
rb=[];
pre=[];
l=[];

for i=1:10
    [w_train,b_train]=CD_Lasso(y,X,lambda);
    rw=[rw,w_train];
    rb=[rb,b_train];
    if(size(find(w_train),1)==0)
        prec=0;
    else
        prec=size(find(abs(w_train(1:10,1)-w(1:10,1))<=0.2),1)/size(find(w_train),1);
    end
    recall=size(find(abs(w_train(1:10,1)-w(1:10,1))<=0.2),1)/10;
    pre(:,i)=[prec;recall];
    l=[l;lambda];
    disp(['finding proper lambda, round: ',num2str(i),', lambda is ',num2str(lambda)]);
    lambda=lambda/2;
end

fig=figure();
plot([1:10],pre(1,:),'-ro',[1:10],pre(2,:),'-xb');
axis([0,10,0,1]);
set(gca,'xticklabel', [0;l])
text(1,0.9,'sigma=10');
legend('Prediction','Recall');
% saveas(fig,'result(simga=10).png');


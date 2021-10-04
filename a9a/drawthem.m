close all;
clc;
clear;
fval=0.2077;
load('data/opti_x.mat');
xstar=x;
%% meth1_sw
% ==============需要读取.mat文件================
  load('data/G_meth1_smote_sw_800.mat');%加载保存的迭代梯度信息x_k_store{}
  load('data/X_meth1_smote_sw_800.mat');%加载保存的迭代解信息gradient()
  load('data/C_meth1_smote_sw_800.mat');%加载保存的邻接矩阵信息C_store
  lamuda1=5*10^(-4);
  lamuda2=10^(-4);
  load('a9a.mat');
  load('L_a9a.mat');
  L=double(L);
  L(L==0)=-1;
  C=C_store{1};
% 参数设置
Maxgen1s=size(x_k_store,2);%迭代次数
agent_num=size(C);%智能体个数

% 训练集
for k=1:Maxgen1s
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  % 目标函数
  fi=sum(1./(1+exp(L'.*A*x_k_1)),1);
  obj_m1s(k)=fi/size(A,1)+lamuda1*norm(x_k_1,1)+lamuda2*norm(x_k_1,2)^2-fval; 
  %  XLX
  XLX=0;
  for i=1:agent_num
    temp_X=zeros(size(A,2),1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(x_k(:,i)-x_k(:,j));
    end
    XLX=XLX+x_k(:,i)'*temp_X;
  end
  w_m1s(k)=XLX;
  % 梯度值
   %取所有智能体中第一个
   g_k=gradient_sto{k}+lamuda1*sign(x_k_1);
   zz_m1s(k)=norm(g_k(:,1));
   
   xx_v1s(k)=norm(x_k_1-xstar);
end
% 测试集
  load('a9a_test.mat');
  load('L_a9a_test.mat');
  L=double(L);
  L(L==0)=-1;
for k=1:Maxgen1s
     % 测试集正确率
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  result=A*x_k_1;
  result(result>=0)=1;
  result(result<0)=-1;

  yy_m1s(k)=sum((result==L'))/size(L,2);
end

%% meth 2
% ==============需要读取.mat文件================
  load('data/G_meth2_smote_sw_800.mat');%加载保存的迭代梯度信息x_k_store{}
  load('data/X_meth2_smote_sw_800.mat');%加载保存的迭代解信息gradient()
  load('data/C_meth1_smote_sw_800.mat');%加载保存的邻接矩阵信息C_store
  lamuda1=5*10^(-4);
  lamuda2=10^(-4);
  load('a9a.mat');
  load('L_a9a.mat');
  L=double(L);
  L(L==0)=-1;
  C=C_store{1};
% 参数设置
Maxgen2s=size(x_k_store,2);%迭代次数
agent_num=size(C);%智能体个数

% 训练集
for k=1:Maxgen2s
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  % 目标函数
  fi=sum(1./(1+exp(L'.*A*x_k_1)),1);
  obj_m2s(k)=fi/size(A,1)+lamuda1*norm(x_k_1,1)+lamuda2*norm(x_k_1,2)^2-fval; 
  %  XLX
  XLX=0;
  for i=1:agent_num
    temp_X=zeros(size(A,2),1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(x_k(:,i)-x_k(:,j));
    end
    XLX=XLX+x_k(:,i)'*temp_X;
  end
  w_m2s(k)=XLX;
  % 梯度值
   %取所有智能体中第一个
   g_k=gradient_sto{k}+lamuda1*sign(x_k_1);
   zz_m2s(k)=norm(g_k(:,1));
   
   xx_v2s(k)=norm(x_k_1-xstar);
end
% 测试集
  load('a9a_test.mat');
  load('L_a9a_test.mat');
  L=double(L);
  L(L==0)=-1;
for k=1:Maxgen2s
     % 测试集正确率
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  result=A*x_k_1;
  result(result>=0)=1;
  result(result<0)=-1;

  yy_m2s(k)=sum((result==L'))/size(L,2);
end
%% meth3
% ==============需要读取.mat文件================
  load('data/G_meth3_smote_sw_800.mat');%加载保存的迭代梯度信息x_k_store{}
  load('data/X_meth3_smote_sw_800.mat');%加载保存的迭代解信息gradient()
  load('data/C_meth1_smote_sw_800.mat');%加载保存的邻接矩阵信息C_store
  lamuda1=5*10^(-4);
  lamuda2=10^(-4);
  load('a9a.mat');
  load('L_a9a.mat');
  L=double(L);
  L(L==0)=-1;
  C=C_store{1};
% 参数设置
Maxgen3s=size(x_k_store,2);%迭代次数
agent_num=size(C);%智能体个数

% 训练集
for k=1:Maxgen3s
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  % 目标函数
  fi=sum(1./(1+exp(L'.*A*x_k_1)),1);
  obj_m3(k)=fi/size(A,1)+lamuda1*norm(x_k_1,1)+lamuda2*norm(x_k_1,2)^2-fval; 
  %  XLX
  XLX=0;
  for i=1:agent_num
    temp_X=zeros(size(A,2),1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(x_k(:,i)-x_k(:,j));
    end
    XLX=XLX+x_k(:,i)'*temp_X;
  end
  w_m3(k)=XLX;
  % 梯度值
   %取所有智能体中第一个
   g_k=gradient_sto{k}+lamuda1*sign(x_k_1);
   zz_m3(k)=norm(g_k(:,1));
   xx_v3(k)=norm(x_k_1-xstar);
end
% 测试集
  load('a9a_test.mat');
  load('L_a9a_test.mat');
  L=double(L);
  L(L==0)=-1;
for k=1:Maxgen3s
     % 测试集正确率
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  result=A*x_k_1;
  result(result>=0)=1;
  result(result<0)=-1;

  yy_m3(k)=sum((result==L'))/size(L,2);
  
end


%% 绘图  
figure(1);
plot(1:120,w_m1s(1:120),'r-.','linewidth',1.2),hold on;
plot(1:4:120,w_m2s(1:4:120),'b-x','linewidth',1),hold on;
plot(1:4:120,w_m3(1:4:120),'k-o','linewidth',1), hold on;
ylim([0 0.01]);
legend('DPG-RR','Prox-G','NEXT');
ylabel('$$\|D(\bar{x})\|$$','Interpreter','latex')
xlabel('iterations');
figure(2);
plot(1:120,obj_m1s(1:120),'r-.','linewidth',1.2),hold on;
plot(1:4:120,obj_m2s(1:4:120),'b-x','linewidth',1),hold on;
plot(1:4:120,obj_m3(1:4:120),'k-o','linewidth',1), hold on;
 legend('DPG-RR','Prox-G','NEXT');
ylabel('$$F(x)-F_*$$','Interpreter','latex')
xlabel('iterations')



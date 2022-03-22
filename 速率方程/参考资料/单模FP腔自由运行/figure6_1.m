clear all;
clf;
tic;
%定义参数
c=3e8;%光速
q=1.6e-19;%电子电荷
lambda=1.3e-6;%波长
L=250e-6;%腔长
w=2e-6;%有源区宽度
d=0.2e-6;%厚度
Gama=0.3;%限制因子
miu=3.4;%有效模式率
miu_g=4;%群反射率
betac=5;%线宽增强因子
alpha_m=45e2;%解理面损耗
alpha_int=40e2;%内部损耗
a=2.5e-20;%增益系数
n0=1e24;%透明载流子密度
Anr=1e8;%非辐射复合率
B=1e-16;%辐射复合率
C=3e-41;%俄歇复合率
Nth=2.14e8;%载流子阈值
Ith=15.8e-3;%电流阈值
tau_e=2.2e-9;%载流子寿命
tau_p=1.6e-12;%光子寿命
h_p=6.62*1e-34;
%%
I=0:0.0001:0.03;
m=length(I);
V=L*w*d;%腔的体积
gama=1/tau_p;
gama_e=1/tau_e;
vg=gama/(alpha_m+alpha_int);%群速度miu=c/lambda;
niu=c/lambda;
T=1/2*h_p*niu*vg*alpha_m;
h=1e-12;%步长
nn=1e5;%取点数
t=0:h:(nn-1)*h;%%总时间
Power=1e-3;
P=zeros(1,nn);
P(1,1)=1e-5;
N=zeros(1,nn);
N(1,1)=1e-3;
PP=zeros(1,m);
NN=zeros(1,m);
P1=Power/T;
P(1,1)=1e-6;
%%
M=[1e-5 1e-4 1e-3];
for ii=1:3;
    beta_sp=M(ii);
    for i=1:m;
        for j=1:nn-1;
            %第一次迭代
            PK=P(1,j);NK=N(1,j);
            PK1=h*((Gama*vg*a*(NK/V-n0)-gama)*PK+beta_sp*B*NK^2/V);
            NK1=h*(I(i)/q-(Anr+B*NK/V+C*(NK/V)^2)*NK-(Gama*vg*a*(NK/V-n0)*PK));
            %第二次迭代
            PK=P(1,j)+1/2*PK1;NK=N(1,j)+1/2*NK1;
            PK2=h*((Gama*vg*a*(NK/V-n0)-gama)*PK+beta_sp*B*NK^2/V);
            NK2=h*(I(i)/q-(Anr+B*NK/V+C*(NK/V)^2)*NK-(Gama*vg*a*(NK/V-n0)*PK));
            %第三次迭代
            PK=P(1,j)+1/2*PK2;NK=N(1,j)+1/2*NK2;
            PK3=h*((Gama*vg*a*(NK/V-n0)-gama)*PK+beta_sp*B*NK^2/V);
            NK3=h*(I(i)/q-(Anr+B*NK/V+C*(NK/V)^2)*NK-(Gama*vg*a*(NK/V-n0)*PK));
            %第四次迭代
            PK=P(1,j)+PK3;NK=N(1,j)+NK3;
            PK4=h*((Gama*vg*a*(NK/V-n0)-gama)*PK+beta_sp*B*NK^2/V);
            NK4=h*(I(i)/q-(Anr+B*NK/V+C*(NK/V)^2)*NK-(Gama*vg*a*(NK/V-n0)*PK));
            %用四阶龙格库塔求值%
             P(1,j+1)=P(1,j)+(PK1+2*PK2+2*PK3+PK4)/6;
             N(1,j+1)=N(1,j)+(NK1+2*NK2+2*NK3+NK4)/6;

        end
    PP(1,i)=(mean(P(1,1e4:1e5)));
    NN(1,i)=(mean(N(1,1e4:1e5)));
%     PP(1,i)=P(100,end);
%     NN(1,i)=N(100,end);
    % Pm=0.5*h_p/(2*pi)*
    end
semilogy(I*1e3,T*PP*1e2,'k','linewidth',2);
axis([0 30 1e-8 10]);
hold on;
end
xlabel('Device Current(mA)');
ylabel('Power(W)');
text(8,1e-4,'\beta_s_p');
text(8,5e-6,'10^-^3');
text(8,5e-7,'10^-^4');
text(8,2e-7,'10^-^5');
text(20,5e-7,'I_t_h');
text(20,2e-7,'\downarrow');
toc;
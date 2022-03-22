clear all;
clf;
tic;
%�������
c=3e8;%����
q=1.6e-19;%���ӵ��
lambda=1.3e-6;%����
L=250e-6;%ǻ��
w=2e-6;%��Դ�����
d=0.2e-6;%���
Gama=0.3;%��������
miu=3.4;%��Чģʽ��
miu_g=4;%Ⱥ������
betac=5;%�߿���ǿ����
alpha_m=45e2;%���������
alpha_int=40e2;%�ڲ����
a=2.5e-20;%����ϵ��
n0=1e24;%͸���������ܶ�
Anr=1e8;%�Ƿ��临����
B=1e-16;%���临����
C=3e-41;%��Ъ������
Nth=2.14e8;%��������ֵ
Ith=15.8e-3;%������ֵ
tau_e=2.2e-9;%����������
tau_p=1.6e-12;%��������
h_p=6.62*1e-34;
%%
I=0:0.0001:0.03;
m=length(I);
V=L*w*d;%ǻ�����
gama=1/tau_p;
gama_e=1/tau_e;
vg=gama/(alpha_m+alpha_int);%Ⱥ�ٶ�miu=c/lambda;
niu=c/lambda;
T=1/2*h_p*niu*vg*alpha_m;
h=1e-12;%����
nn=1e5;%ȡ����
t=0:h:(nn-1)*h;%%��ʱ��
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
            %��һ�ε���
            PK=P(1,j);NK=N(1,j);
            PK1=h*((Gama*vg*a*(NK/V-n0)-gama)*PK+beta_sp*B*NK^2/V);
            NK1=h*(I(i)/q-(Anr+B*NK/V+C*(NK/V)^2)*NK-(Gama*vg*a*(NK/V-n0)*PK));
            %�ڶ��ε���
            PK=P(1,j)+1/2*PK1;NK=N(1,j)+1/2*NK1;
            PK2=h*((Gama*vg*a*(NK/V-n0)-gama)*PK+beta_sp*B*NK^2/V);
            NK2=h*(I(i)/q-(Anr+B*NK/V+C*(NK/V)^2)*NK-(Gama*vg*a*(NK/V-n0)*PK));
            %�����ε���
            PK=P(1,j)+1/2*PK2;NK=N(1,j)+1/2*NK2;
            PK3=h*((Gama*vg*a*(NK/V-n0)-gama)*PK+beta_sp*B*NK^2/V);
            NK3=h*(I(i)/q-(Anr+B*NK/V+C*(NK/V)^2)*NK-(Gama*vg*a*(NK/V-n0)*PK));
            %���Ĵε���
            PK=P(1,j)+PK3;NK=N(1,j)+NK3;
            PK4=h*((Gama*vg*a*(NK/V-n0)-gama)*PK+beta_sp*B*NK^2/V);
            NK4=h*(I(i)/q-(Anr+B*NK/V+C*(NK/V)^2)*NK-(Gama*vg*a*(NK/V-n0)*PK));
            %���Ľ����������ֵ%
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
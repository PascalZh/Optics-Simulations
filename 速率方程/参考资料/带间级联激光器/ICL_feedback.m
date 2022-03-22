%%%%%���Ӽ���������ICL�ڲ�ͬ�ⷴ��ǿ�������µ����%%%%%%%%%%%%
clear all;
%close all;
clc;
%%
%ʱ�䴰��
h=1e-13;  %����
nn=2^23;  %ȡ����
tt=h*(nn-1);  %��ʱ�䴰��
t=0:h:tt;  %ʱ�䴰�ڵ�ʱ���ֲ�
n=length(t);  %ʱ��ȡ����Ŀ
fs=1/h;
%%
%��������֪����ֵ
H=0.64;  %ע�����Ч��
q=1.6*10^(-19); %���ӵ��(c)
c=3*10^8;  %����(m/s)
lamda=3.7*10^(-6);  %����(m)
L=2*10^-3;  %ǻ��(m)
W=4.4*10^(-6);  %ǻ��(m)
n_r=3.58;  %������
m=5; %���Ӽ�����
gamma_p=0.04;  %����������
v_g=8.38*10^7;  %Ⱥ�ٶ�(m/s)
a0=2.8*10^(-12);  %����
Ntr=6.2*10^7;   %��ֵ��������
A=8.8*10^(-9);  %��Դ�����(m^2)
tau_sp=15*10^(-9);  %�Է���������(s)
tau_aug=1.08*10^(-9);  %Auger����(s)
tau_p=10.5*10^(-12);  %��������(s)
tau_in=2*L/c*n_r;  %ǻ������ʱ��(s)
tau_f=1.44*10^(-9);  %�ⷴ���ӳ�ʱ��(s)
w=2*pi*c/lamda;  %��Ľ�Ƶ��
beta=1*10^(-4);  %�Է��������ӣ�����
R=0.32; %���侵������
Ith=q/H*(1/m*A/(gamma_p*v_g*a0*tau_p)+Ntr)*(1/tau_sp+1/tau_aug);  %��ֵ����
I=1.01*Ith;          %ע�����
C_l=(1-R)/(2*sqrt(R));  %�ⲿ���ϵ��
alfa_H=2.2;  %�߿���ǿ����
delayNum=round(tau_f/h);  %��ʱtau_f�ȼ۵ĵ���
%%
%�洢����ͳ�ʼֵ
N=zeros(1,n);            %���������ܶ�
S=zeros(1,n);             %�������ܶ�
Fi=zeros(1,n);            %�糡����λ
N(1)=0.5*Ntr;  %���������ܶȳ�ʼֵ
S(1)=1;     %�������ܶ�
%%
% %��ͬ����ǿ�ȶ������Ӱ��
f_ext_array=[0.0 3 0.14 0.2 0.84]*0.01;  %������ǿ�ͼ����������ǿ��ֵ����
f_ext=f_ext_array(2);  %������ǿ�ͼ����������ǿ��ֵ
kap=2*C_l*sqrt(f_ext)/tau_in;  %����ǿ��
%%
%�Ľ����������ʽ
for ii=1:n-1 
    if ii>delayNum
        key=1;mm=ii-delayNum;
    else
        key=0;mm=1;
    end
    %%%%%%%��һ�ε���
    Nt=N(ii);St=S(ii);Fit=Fi(ii);
    Nk1=H*I/q-gamma_p*v_g*a0*(Nt-Ntr)/A*St-Nt/tau_sp-Nt/tau_aug;
    Sk1=(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)*St+m*beta*Nt/tau_sp+...
        2*kap*key*sqrt(St*S(mm))*cos(w*tau_f+Fit-Fi(mm));
    Fik1=alfa_H/2*(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)-key*kap*sqrt(S(mm)/St)*sin(w*tau_f+Fit-Fi(mm));
    
    %%%%%%%�ڶ��ε���
    Nt=N(ii)+h*Nk1/2; St=S(ii)+Sk1*h/2;Fit=Fi(ii)+h*Fik1/2;
    Nk2=H*I/q-gamma_p*v_g*a0*(Nt-Ntr)/A*St-Nt/tau_sp-Nt/tau_aug;
    Sk2=(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)*St+m*beta*Nt/tau_sp+...
        2*kap*key*sqrt(St*S(mm))*cos(w*tau_f+Fit-Fi(mm));
    Fik2=alfa_H/2*(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)-key*kap*sqrt(S(mm)/St)*sin(w*tau_f+Fit-Fi(mm));
    
    %%%%%%%�����ε���
    Nt=N(ii)+h*Nk2/2; St=S(ii)+Sk2*h/2;Fit=Fi(ii)+h*Fik2/2;
    Nk3=H*I/q-gamma_p*v_g*a0*(Nt-Ntr)/A*St-Nt/tau_sp-Nt/tau_aug;
    Sk3=(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)*St+m*beta*Nt/tau_sp+...
        2*kap*key*sqrt(St*S(mm))*cos(w*tau_f+Fit-Fi(mm));
    Fik3=alfa_H/2*(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)-key*kap*sqrt(S(mm)/St)*sin(w*tau_f+Fit-Fi(mm));
    
    %%%%%%%���Ĵε���
    Nt=N(ii)+h*Nk3; St=S(ii)+Sk3*h;Fit=Fi(ii)+h*Fik3;
    Nk4=H*I/q-gamma_p*v_g*a0*(Nt-Ntr)/A*St-Nt/tau_sp-Nt/tau_aug;
    Sk4=(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)*St+m*beta*Nt/tau_sp+...
        2*kap*key*sqrt(St*S(mm))*cos(w*tau_f+Fit-Fi(mm));
    Fik4=alfa_H/2*(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)-key*kap*sqrt(S(mm)/St)*sin(w*tau_f+Fit-Fi(mm));
    
    %%%%%%%��һ�㹫ʽ
    N(ii+1)=N(ii)+h*(Nk1+2*Nk2+2*Nk3+Nk4)/6;
    S(ii+1)=S(ii)+h*(Sk1+2*Sk2+2*Sk3+Sk4)/6;
    Fi(ii+1)=Fi(ii)+h*(Fik1+2*Fik2+2*Fik3+Fik4)/6;
end

%%
%��ͼ����
figure(1)
%%%%%%%%%%%��ʱ������
 subplot(2,2,1)
 T=t*10^9;
  plot(T(end-2^22+1:end),S(end-2^22+1:end))
  xlabel('Time[ns]');
  ylabel('Photon number');
  title('TS');
  axis([450,600,-0.1e5,1.4e5])
%%%%%%%%%%%%%����ͼ
  subplot(2,2,2)
  plot(N(end-2^22+1:end),S(end-2^22+1:end),'r')
  xlabel('Carrier number');
  ylabel('Photon number');
  title('PP');
  %axis([7.95e7,8.05e7,0e5,2e5])
%%%%%%%%%%%%��������
 I1=S(end-2^22+1:end);
Fs=1/h;
NN=length(I1);
Y=abs(fft(I1)).^2/(NN/2);
Y(1)=Y(1)/2;
WA_esa=10*log10(Y(1:NN/2));
freq=(0:NN-1)*Fs/NN;
freq_esa=freq(1:NN/2)*1e-9;
subplot(2,2,3)
plot(freq_esa,WA_esa,'r')
xlabel('Frequency��GHz��');
ylabel('Magnitude[dB]');
axis([0,4,0,200])
PPP=max (WA_esa)
%%  lyapunov ex
% xdata =S(end-2^22+1:end) ;
% dim = 3;
%  [~,lag] = phaseSpaceReconstruction(xdata,[],dim)
%  lag=10;
%  eRange = 200;
% lyapunovExponent(xdata,fs,'Dimension',dim,'Lag',lag,'ExpansionRange',eRange)
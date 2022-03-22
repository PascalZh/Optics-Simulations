%%%%%量子级联激光器ICL在不同光反馈强度作用下的输出%%%%%%%%%%%%
clear all;
%close all;
clc;
%%
%时间窗口
h=1e-13;  %步长
nn=2^23;  %取点数
tt=h*(nn-1);  %总时间窗口
t=0:h:tt;  %时间窗口的时间点分布
n=length(t);  %时间取点数目
fs=1/h;
%%
%方程中已知参数值
H=0.64;  %注入电流效率
q=1.6*10^(-19); %电子电荷(c)
c=3*10^8;  %光速(m/s)
lamda=3.7*10^(-6);  %波长(m)
L=2*10^-3;  %腔长(m)
W=4.4*10^(-6);  %腔宽(m)
n_r=3.58;  %折射率
m=5; %量子级联数
gamma_p=0.04;  %光限制因子
v_g=8.38*10^7;  %群速度(m/s)
a0=2.8*10^(-12);  %增益
Ntr=6.2*10^7;   %阈值载流子数
A=8.8*10^(-9);  %有源区面积(m^2)
tau_sp=15*10^(-9);  %自发辐射寿命(s)
tau_aug=1.08*10^(-9);  %Auger寿命(s)
tau_p=10.5*10^(-12);  %光子寿命(s)
tau_in=2*L/c*n_r;  %腔内往返时间(s)
tau_f=1.44*10^(-9);  %光反馈延迟时间(s)
w=2*pi*c/lamda;  %光的角频率
beta=1*10^(-4);  %自发发射因子（？）
R=0.32; %反射镜反射率
Ith=q/H*(1/m*A/(gamma_p*v_g*a0*tau_p)+Ntr)*(1/tau_sp+1/tau_aug);  %阈值电流
I=1.01*Ith;          %注入电流
C_l=(1-R)/(2*sqrt(R));  %外部耦合系数
alfa_H=2.2;  %线宽增强因子
delayNum=round(tau_f/h);  %延时tau_f等价的点数
%%
%存储矩阵和初始值
N=zeros(1,n);            %载流子数密度
S=zeros(1,n);             %光子数密度
Fi=zeros(1,n);            %电场的相位
N(1)=0.5*Ntr;  %载流子数密度初始值
S(1)=1;     %光子数密度
%%
% %不同反馈强度对输出的影响
f_ext_array=[0.0 3 0.14 0.2 0.84]*0.01;  %反馈光强和激光器输出光强比值区间
f_ext=f_ext_array(2);  %反馈光强和激光器输出光强比值
kap=2*C_l*sqrt(f_ext)/tau_in;  %反馈强度
%%
%四阶龙格库塔公式
for ii=1:n-1 
    if ii>delayNum
        key=1;mm=ii-delayNum;
    else
        key=0;mm=1;
    end
    %%%%%%%第一次迭代
    Nt=N(ii);St=S(ii);Fit=Fi(ii);
    Nk1=H*I/q-gamma_p*v_g*a0*(Nt-Ntr)/A*St-Nt/tau_sp-Nt/tau_aug;
    Sk1=(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)*St+m*beta*Nt/tau_sp+...
        2*kap*key*sqrt(St*S(mm))*cos(w*tau_f+Fit-Fi(mm));
    Fik1=alfa_H/2*(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)-key*kap*sqrt(S(mm)/St)*sin(w*tau_f+Fit-Fi(mm));
    
    %%%%%%%第二次迭代
    Nt=N(ii)+h*Nk1/2; St=S(ii)+Sk1*h/2;Fit=Fi(ii)+h*Fik1/2;
    Nk2=H*I/q-gamma_p*v_g*a0*(Nt-Ntr)/A*St-Nt/tau_sp-Nt/tau_aug;
    Sk2=(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)*St+m*beta*Nt/tau_sp+...
        2*kap*key*sqrt(St*S(mm))*cos(w*tau_f+Fit-Fi(mm));
    Fik2=alfa_H/2*(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)-key*kap*sqrt(S(mm)/St)*sin(w*tau_f+Fit-Fi(mm));
    
    %%%%%%%第三次迭代
    Nt=N(ii)+h*Nk2/2; St=S(ii)+Sk2*h/2;Fit=Fi(ii)+h*Fik2/2;
    Nk3=H*I/q-gamma_p*v_g*a0*(Nt-Ntr)/A*St-Nt/tau_sp-Nt/tau_aug;
    Sk3=(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)*St+m*beta*Nt/tau_sp+...
        2*kap*key*sqrt(St*S(mm))*cos(w*tau_f+Fit-Fi(mm));
    Fik3=alfa_H/2*(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)-key*kap*sqrt(S(mm)/St)*sin(w*tau_f+Fit-Fi(mm));
    
    %%%%%%%第四次迭代
    Nt=N(ii)+h*Nk3; St=S(ii)+Sk3*h;Fit=Fi(ii)+h*Fik3;
    Nk4=H*I/q-gamma_p*v_g*a0*(Nt-Ntr)/A*St-Nt/tau_sp-Nt/tau_aug;
    Sk4=(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)*St+m*beta*Nt/tau_sp+...
        2*kap*key*sqrt(St*S(mm))*cos(w*tau_f+Fit-Fi(mm));
    Fik4=alfa_H/2*(m*gamma_p*v_g*a0*(Nt-Ntr)/A-1/tau_p)-key*kap*sqrt(S(mm)/St)*sin(w*tau_f+Fit-Fi(mm));
    
    %%%%%%%下一点公式
    N(ii+1)=N(ii)+h*(Nk1+2*Nk2+2*Nk3+Nk4)/6;
    S(ii+1)=S(ii)+h*(Sk1+2*Sk2+2*Sk3+Sk4)/6;
    Fi(ii+1)=Fi(ii)+h*(Fik1+2*Fik2+2*Fik3+Fik4)/6;
end

%%
%画图部分
figure(1)
%%%%%%%%%%%画时间序列
 subplot(2,2,1)
 T=t*10^9;
  plot(T(end-2^22+1:end),S(end-2^22+1:end))
  xlabel('Time[ns]');
  ylabel('Photon number');
  title('TS');
  axis([450,600,-0.1e5,1.4e5])
%%%%%%%%%%%%%画相图
  subplot(2,2,2)
  plot(N(end-2^22+1:end),S(end-2^22+1:end),'r')
  xlabel('Carrier number');
  ylabel('Photon number');
  title('PP');
  %axis([7.95e7,8.05e7,0e5,2e5])
%%%%%%%%%%%%画功率谱
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
xlabel('Frequency（GHz）');
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
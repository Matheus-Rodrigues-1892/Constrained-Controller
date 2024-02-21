%System: x(k + 1) = Ax(k) + Bu(k) + Ed(k)
%            y(k) = Cx(k) + eta(k)
clear all; clc;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
% Double-Tank Process %
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Tank 1 and Tank 2 inside diameter (cm)
D1 = 4.445; D2 = 4.445;
% Tank 1 and Tank 2 inside area (cm^2)
A1 = (pi/4)*D1^2; A2 = (pi/4)*D2^2;
% Out 1 and Out 2 orifice diameter (cm)
d1 = 0.476; d2 = 0.476;
% Out 1 and Out 2 orifice area (cm^2)
a1 = (pi/4)*d1^2; a2 = (pi/4)*d2^2;
% Pump flow constant (cm^3/Vs)
km = 4.1; %4.2603; %3.3 
% Tank 1 and tank 2 water level sensor sensitivity (V/cm)
% (depending on the pressure sensor calibration)
kc = 1; %6.1;
% Gravitational Constant on Earth (cm/s^2)
g = 981;

%% Definimos x1 = h1 e x2 = h2
%  Cálculo do ueq (tensão na bomba) em função do x1eq
x1eq = 15;
ueq = (a1/km)*sqrt(2*g*x1eq);
%% Parameter values corresponding to the operating points
%  Para um dado valor de 'ueq'

x1eq = (1/(2*g))*((km*ueq)/a1)^2;
x2eq = x1eq*(a1/a2)^2;

%% The measured level signals are kc*x1 and kc*x2. 

%% Linearized System Model

Ac = [(-a1/A1)*sqrt(g/(2*x1eq)) 0; (a1/A2)*sqrt(g/(2*x1eq)) (-a2/A2)*sqrt(g/(2*x2eq))];
Bc = [km/A1; 0];
Cc = [0 1];
Dc = 0;

sys = ss(Ac,Bc,Cc,Dc);
Ts = 0.5;
sysd = c2d(sys,Ts);
A = sysd.A; B = sysd.B; C = sysd.C; E = [0 0]'; 
%E = zeros(4,2);

%%
%%%%%%%%%%%%%%%
% Constraints %
%%%%%%%%%%%%%%%

G = [1 0; -1 0; 0 1; 0 -1];
rho = [(30-x1eq) x1eq (30-x2eq) x2eq]';
U = [1 -1]';
phi = [(12-ueq) ueq]';
D = [1 -1]';
w = [1 1]';
N = [1 -1]';
etab = [0 0]';
lambda_v = 0.99;

%Px = Polyhedron(G,rho); plot(Px)

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Controlled-Invariant polyhedron with contraction rate "lambda_v" %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Gv,rhov,~] = maxdelduabinvdisc(A,B,E,G,rho,D,w,U,phi,lambda_v);
hold on
Pv = Polyhedron(Gv,rhov); plot(Pv)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fixed parameters for the simulations %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

it = 700;

%Perturbação aleatória
d1 = randn(1,it);
d1 = 0.001*inv(max(abs(d1)))*d1;

%Perturbação aleatória
d2 = randn(1,it);
d2 = 0.001*inv(max(abs(d2)))*d2;

%Ruído aleatório
n1 = randn(1,it);
n1 = 0.1*inv(max(abs(n1)))*n1;

%Ruído aleatório
n2 = randn(1,it);
n2 = 0.1*inv(max(abs(n2)))*n2;

%Pior caso para a perturbação em relação ao sistema
delta = maxdistvect(E,Gv,D,w); 

%Bola em torno da origem
Gi = [eye(nla);-eye(nla)];
rhoi = 14.8456*ones(2*nla,1);

%Pior caso para a perturbação em relação à bola em torno da origem
deltai = maxdistvect(E,Gi,D,w);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STATIC STATE FEEDBACK CONTROLLER %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xk = [-5 -20]';
X = xk;
yk = C*xk;
Y = yk;
uk = zeros(it,1);
t = zeros(it+1,1);
t(1) = 0;
epsi_LP = [];
[nlu, ncu] = size(U);
% O loop de simulação sumirá dando lugar ao sistema real
for k = 1:it
    %xk = [H1 H2]';
    fobj = [zeros(ncu, 1); 1];   
    Apo = [Gv*B -rhov; U zeros(nlu, 1)];
    Bpo = [-Gv*A*xk; phi];
    
    %       arg    min     epsi
    %            u,epsi
    %
    %         |Gv*B  -rhov|       | -Gv*A*xk |
    % Apo   = |  U     0  | Bpo = |    phi   |
    
    v = linprog(fobj, Apo, Bpo);
    uk(k,:) = v(1);%Será mandado para o simulink
    epsi_LP = [epsi_LP v(2)];

    %As duas equqções simulam o sistema
    xk = A*xk + B*uk(k,:);
    yk = C*xk;

X = [X xk];
Y = [Y yk];
t(k+1) = k;
end

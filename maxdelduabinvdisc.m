function [Gf,rhof,it]=maxdelduabinvdisc(aa,bb,e,G,rho,D,omega,U,phi,lambda)

%function [Gf,rhof,it]=maxdelduabinvdisc(aa,bb,e,G,rho,D,omega,U,phi,lambda)
%
% Computation of the maximal Delta-D-U-(a,b)-invariant set, with a
% contraction rate lambda :
% 	Gf*x <= rhof
% contained in :  Gx <= rho
%
% for a system subject to bounded disturbances:
%	Dd <= omega
% and to control constrints:
%   Uu<=phi
%
% it = number of iterations

[g,n]=size(G);
[d,p]=size(D);
[na,n]=size(aa);na=na/n;
[nb,m]=size(bb);nb=nb/n;

Gi=G;rhoi=rho;
it=0;
matti=1; %lmatti=0;
while any(matti) && it<=1000;
  it=it+1;
    [rGi,cGi]=size(Gi);rGi
%  if lmatti~=0
%    mattiaGi=(abs(ti(:,1:rGi-lmatti)*Gi(1:rGi-lmatti,:)*b)>1e-8);
%  else
%    [rti,cti]=size(ti);
%    mattiaGi=ones(rti,1);
%  end
%  ti=ti(mattiaGi,:);
  vi=maxdistvect(e,Gi,D,omega);
  vvi=kron(ones(na*nb,1),vi);
  vvi=[vvi;zeros(size(U,1),1)];
  [yi,tgi,gaa0i,gbbi,rhogi]=deluabinvdisc(aa,bb,Gi,rhoi,U,phi);
  matti=((yi*rhoi-tgi*[lambda*rhoi-vi;phi])>1e-5);
  Gi=[Gi;tgi(matti,:)*gaa0i];
  rhoi=[rhoi;tgi(matti,:)*[lambda*rhoi-vi;phi]];
%  lmatti=length(ti(matti,1));

%   Pi=polytope(Gi,rhoi);
%   [Gi,rhoi]=double(Pi);
  [Gi,rhoi,lgirhoi]=elimredgro(Gi,rhoi);

%  if min(lgirhoi)<=rGi
%    lmatti=0;
%  end
end
[Gf,rhof,leg]=elimredgro(Gi,rhoi);
% Pf=polytope(Gi,rhoi);
% [Gf,rhof]=double(Pf);
end

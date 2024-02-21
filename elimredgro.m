function [Gn,ron,le]=elimredgro(G,ro)

%function [Gn,ron,le]=elimredgro(G,ro);
%
% given G,ro find:
%
% le --> redundant rows
% Gn,ron --> G,ro without the redundant rows

[g,n]=size(G);

le=[];lne=[1:g]';
for i=1:g
  Gc=G;roc=ro;
  Gc([i;le],:)=[];roc([i;le])=[];
%  x=linprog(-G(i,:),Gc,roc);
   IN.A=[Gc];
   IN.B=[roc];
   IN.obj=-G(i,:);
   OUT=cddmex('solve_lp',IN);
   x=OUT.xopt;
  if G(i,:)*x<=(1+1e-4)*ro(i)
    le=[le;i];
    lne(i-length(le)+1)=[];
  end
end

Gn=G(lne,:);
ron=ro(lne);
end


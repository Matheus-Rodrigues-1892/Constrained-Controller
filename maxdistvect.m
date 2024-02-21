function v=maxdistvect(e,G,D,w)

%function v=maxdistvect(e,G,D,w)
%
%	Compute v(i)=max G(i,:)*e*q
%		      q
%
%		s.t.   Dq <= w

[g,n]=size(G);
for i=1:g
%  qi=linprog(-G(i,:)*e,D,w);
%   v(i,1)=G(i,:)*e*qi;
  
IN.A=[D];
IN.B=[w];
IN.obj=(-G(i,:)*e);

OUT=cddmex('solve_lp',IN);
v(i,1)=-OUT.objlp;

end

end

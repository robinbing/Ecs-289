function [U, D] = inductive_mc(X, A, kx, ncluster, lambda, maxit)

A = sparse(A);
n = size(X,1);
[UX SX VX] = svds(X, kx);
U = zeros(kx,1);
maxiter = maxit;
D = zeros(1,1);

for iter = 1:maxiter
%	V = U*D;
	if (iter ==1)
		R = CompResidual(A,UX', (UX*U*D*U')');
	end
	grad = -UX'*R*UX;
	[uu ss vv] = svd(U*D*U'-grad);
	U = uu;
	D = diag(max(diag(ss)-lambda,0));
%	fprintf('rank after thresholding: %g\n', nnz(diag(D)));
	for i=ncluster+1:kx
		D(i,i) = 0;
	end

	R = CompResidual(A, UX', (UX*U*D*U')');
	obj = norm(R,'fro')^2+lambda*sum(abs(diag(D)));
%	fprintf('Iter %g obj %g\n', iter, obj);
end

U = UX*U;

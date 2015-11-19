[yorigin, Xorigin] = libsvmread('mushrooms');
rand('seed',0);
randn('seed',0);
aa = randperm(numel(yorigin));

ny1 = nnz(yorigin==1);
ny2 = nnz(yorigin==2);
n = ny1+ny2;
y = [yorigin(yorigin==1); yorigin(yorigin==2)];
X = [Xorigin(yorigin==1,:); Xorigin(yorigin==2,:)];
realA = [ones(ny1,ny1), -ones(ny1,ny2); -ones(ny2, ny1), ones(ny2,ny2)];
rlist = [0.001 0.00001 0.000005 0.000001];
groundtruth = [ones(ny1,ny1) -ones(ny1,ny2); -ones(ny2,ny1) ones(ny2,ny2)];
for rr=1:numel(rlist)
	rate = rlist(rr);
	fprintf('sample rate: %g\n', rate);
	subsp = randperm((ny1+ny2)*(ny1+ny2));
	subsp = subsp(1:floor((ny1+ny2)*(ny1+ny2)*rate));
	A = zeros(ny1+ny2);
	A(subsp) = groundtruth(subsp);
	A = sign(A+A');

	kx = 50;
	ncluster = 2;
	lambda=0.00001;
	[U, D] = inductive_mc(X, A, kx, ncluster, lambda, 50);

	Xresult = U(:,1:ncluster);
	Xresult = Xresult./repmat(sqrt(sum(Xresult.^2,2)), 1, ncluster);
	[idx] = kmeans(Xresult,ncluster);
	predictA = -ones(n,n);
	for i=1:ncluster
		predictA(idx==i, idx==i) = 1;
	end
	accbias = nnz(predictA~=realA)/numel(realA);
	fprintf('Semisupervised Clustering: err %g\n', accbias);
end

% ECE 273 final project
%% Part I. hard margin SVM
%create linear seperate data
% class one center (5,5), class two center (1,1)
data1 = ones(100, 2);
data1 = data1 * 5 + randn(100, 2);

data2 = ones(100, 2);
data2 = data2 + randn(100, 2);
input = [data1; data2];

labels = [ones(100, 1); ones(100, 1) * (-1)];

figure;
h(1:2) = gscatter(input(:,1), input(:,2), labels,'rb','.');

di = 2;
n = 200;
% hard margin SVM training
cvx_begin
    variables w(di) b
    dual variable alphatrain
    minimize(0.5 * norm(w))
    subject to
        labels .* (input * w + b) - 1 >= 0  :alphatrain;
cvx_end

%use grid to plot the figure
% Predict scores over the grid
X = input;

d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
scores = xGrid * w + b;

figure;
h(1:2) = gscatter(X(:,1), X(:,2), labels,'rb','.');
hold on
contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),[0 0],'k');
title('hard margin SVM')
axis equal
hold off

%% Part II. soft margin SVM

data1 = ones(100, 2);
data1 = data1 * 3 + randn(100, 2);

data2 = ones(100, 2);
data2 = data2 + randn(100, 2);
input = [data1; data2];

labels = [ones(100, 1); ones(100, 1) * (-1)];

% Set C before using soft margin SVM
di = 2;
C = 1000;
n = 200;
% soft margin SVM training
cvx_begin
    variables w(di) e(n) b
    dual variable alphatrain
    minimize(0.5 * norm(w) + C * sum(e))
    subject to
        labels .* (input * w + b) - 1 + e >= 0  :alphatrain;
        e >= 0;
cvx_end

% test the result
%use grid to plot the figure
% Predict scores over the grid
X = input;

d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
scores = xGrid * w + b;

figure;
h(1:2) = gscatter(X(:,1), X(:,2), labels,'rb','.');
hold on;
gscatter(X((e>0.001),1), X((e>0.001),2), labels((e>0.001)),'k','o');
hold on
contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),[0 0],'k');
contour(x1Grid,x2Grid,reshape(scores - 1,size(x1Grid)),[0 0],'b');
contour(x1Grid,x2Grid,reshape(scores + 1,size(x1Grid)),[0 0],'r');
title('hard margin SVM')
axis equal
hold off

%% Part III, Kernel trick
sigma = 1;
% for lineaer
K = zeros(n, n);
for i = 1:n
    for j = 1:n
        K(i, j) = exp(-1 * sum((X(i, :) - X(j, :)).^2) / (2*(sigma^2)));
    end
end

cvx_begin
    variable alphatrain(n);
    maximize ((ones(n, 1)'* alphatrain) - 0.5.*quad_form(Y.*alphatrain, K))
    subject to
        alphatrain>=0
        alphatrain<=C
        sum(Y.*alphatrain)==0
cvx_end

alphatrain(alphatrain < 1e-5) = 0;
Xs = X(alphatrain > 0, :);
Ys = Y(alphatrain > 0, :);
alphatrain = alphatrain(alphatrain > 0);

ns = size(alphatrain, 1);
K2 = zeros(n, ns);
for i = 1:n
    for j = 1:ns
        K2(i, j) = exp(-1 * sum((X(i, :) - Xs(j, :)).^2) / (2*(sigma^2)));
    end
end
b = mean(Y - K2*(alphatrain.*Ys));

% predict of training set
Ks = zeros(n, ns);
for i = 1:n
    for j = 1:ns
        Ks(i, j) = exp(-1 * sum((X(i, :) - Xs(j, :)).^2) / (2*(sigma^2)));
    end
end
scores = Ks*(alphatrain.*Ys) - b;

% predict of grids
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

len = size(xGrid, 1);
Ks = zeros(len, ns);
for i = 1:len
    for j = 1:ns
        Ks(i, j) = exp(-1 * sum((xGrid(i, :) - Xs(j, :)).^2) / (2*(sigma^2)));
    end
end

scores = Ks*(alphatrain.*Ys) + b;

figure;
h(1:2) = gscatter(X(:,1), X(:,2), labels,'rb','.');
hold on
contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),[0 0],'k');
title('hard margin SVM')
axis equal
hold off  


%% another datset
data1 = ones(100, 2);
data1 = data1 * 3 + randn(100, 2);

data2 = ones(50, 2);
data2 = data2 + randn(50, 2);

data3 = ones(50, 2);
data3 = data3 * 5 + randn(50, 2);

input = [data1; data2; data3];

labels = [ones(100, 1); ones(100, 1) * (-1)];

figure;
h(1:2) = gscatter(input(:,1), input(:,2), labels,'rb','.');

X = input;
Y = labels;

sigma = 1;
% for lineaer
K = zeros(n, n);
for i = 1:n
    for j = 1:n
        K(i, j) = exp(-1 * sum((X(i, :) - X(j, :)).^2) / (2*(sigma^2)));
    end
end

cvx_begin
    variable alphatrain(n);
    maximize ((ones(n, 1)'* alphatrain) - 0.5.*quad_form(Y.*alphatrain, K))
    subject to
        alphatrain>=0
        alphatrain<=C
        sum(Y.*alphatrain)==0
cvx_end

alphatrain(alphatrain < 1e-5) = 0;
Xs = X(alphatrain > 0, :);
Ys = Y(alphatrain > 0, :);
alphatrain = alphatrain(alphatrain > 0);

ns = size(alphatrain, 1);
K2 = zeros(n, ns);
for i = 1:n
    for j = 1:ns
        K2(i, j) = exp(-1 * sum((X(i, :) - Xs(j, :)).^2) / (2*(sigma^2)));
    end
end
b = mean(Y - K2*(alphatrain.*Ys));

% predict of training set
Ks = zeros(n, ns);
for i = 1:n
    for j = 1:ns
        Ks(i, j) = exp(-1 * sum((X(i, :) - Xs(j, :)).^2) / (2*(sigma^2)));
    end
end
scores = Ks*(alphatrain.*Ys) - b;

% predict of grids
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

len = size(xGrid, 1);
Ks = zeros(len, ns);
for i = 1:len
    for j = 1:ns
        Ks(i, j) = exp(-1 * sum((xGrid(i, :) - Xs(j, :)).^2) / (2*(sigma^2)));
    end
end

scores = Ks*(alphatrain.*Ys) + b;

figure;
h(1:2) = gscatter(X(:,1), X(:,2), labels,'rb','.');
hold on
contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),[0 0],'k');
contour(x1Grid,x2Grid,reshape(scores-1,size(x1Grid)),[0 0],'b');
contour(x1Grid,x2Grid,reshape(scores+1,size(x1Grid)),[0 0],'r');
title('hard margin SVM')
axis equal
hold off  

%% ploy 3
X = input;
Y = labels;
sigma = 1;
% for lineaer
K = zeros(n, n);
for i = 1:n
    for j = 1:n
        K(i, j) = (X(i, :) * X(j, :)')^3;
    end
end

cvx_begin
    variable alphatrain(n);
    maximize ((ones(n, 1)'* alphatrain) - 0.5.*quad_form(Y.*alphatrain, K))
    subject to
        alphatrain>=0
        alphatrain<=C
        sum(Y.*alphatrain)==0
cvx_end

alphatrain(alphatrain < 1e-5) = 0;
Xs = X(alphatrain > 0, :);
Ys = Y(alphatrain > 0, :);
alphatrain = alphatrain(alphatrain > 0);

n = 200;
ns = size(alphatrain, 1);
K2 = zeros(n, ns);
for i = 1:n
    for j = 1:ns
        K2(i, j) = (X(i, :) * Xs(j, :)')^3;
    end
end
b = mean(Y - K2*(alphatrain.*Ys));

% predict of training set
Ks = zeros(n, ns);
for i = 1:n
    for j = 1:ns
        Ks(i, j) = (X(i, :) * Xs(j, :)')^3;
    end
end
scores = Ks*(alphatrain.*Ys) - b;

% predict of grids
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

len = size(xGrid, 1);
Ks = zeros(len, ns);
for i = 1:len
    for j = 1:ns
        Ks(i, j) = (xGrid(i, :) * Xs(j, :)')^3;
    end
end

scores = Ks*(alphatrain.*Ys) + b;

figure;
h(1:2) = gscatter(X(:,1), X(:,2), labels,'rb','.');
hold on
contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),[0 0],'k');
contour(x1Grid,x2Grid,reshape(scores-1,size(x1Grid)),[0 0],'b');
contour(x1Grid,x2Grid,reshape(scores+1,size(x1Grid)),[0 0],'r');
title('hard margin SVM')
axis equal
hold off
%% rbf gamma = 0.01
X = input;
Y = labels;

sigma = 1;
% for lineaer
K = zeros(n, n);
for i = 1:n
    for j = 1:n
        K(i, j) = exp(-0.01 *sum((X(i, :) - X(j, :)).^2) / (2*(sigma^2)));
    end
end

cvx_begin
    variable alphatrain(n);
    maximize ((ones(n, 1)'* alphatrain) - 0.5.*quad_form(Y.*alphatrain, K))
    subject to
        alphatrain>=0
        alphatrain<=C
        sum(Y.*alphatrain)==0
cvx_end

alphatrain(alphatrain < 1e-5) = 0;
Xs = X(alphatrain > 0, :);
Ys = Y(alphatrain > 0, :);
alphatrain = alphatrain(alphatrain > 0);

ns = size(alphatrain, 1);
K2 = zeros(n, ns);
for i = 1:n
    for j = 1:ns
        K2(i, j) = exp(-0.01 * sum((X(i, :) - Xs(j, :)).^2) / (2*(sigma^2)));
    end
end
b = mean(Y - K2*(alphatrain.*Ys));

% predict of training set
Ks = zeros(n, ns);
for i = 1:n
    for j = 1:ns
        Ks(i, j) = exp(-0.01 * sum((X(i, :) - Xs(j, :)).^2) / (2*(sigma^2)));
    end
end
scores = Ks*(alphatrain.*Ys) - b;

% predict of grids
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

len = size(xGrid, 1);
Ks = zeros(len, ns);
for i = 1:len
    for j = 1:ns
        Ks(i, j) = exp(-0.01 * sum((xGrid(i, :) - Xs(j, :)).^2) / (2*(sigma^2)));
    end
end

scores = Ks*(alphatrain.*Ys) + b;

figure;
h(1:2) = gscatter(X(:,1), X(:,2), labels,'rb','.');
hold on
contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),[0 0],'k');
contour(x1Grid,x2Grid,reshape(scores-1,size(x1Grid)),[0 0],'b');
contour(x1Grid,x2Grid,reshape(scores+1,size(x1Grid)),[0 0],'r');
title('hard margin SVM')
axis equal
hold off  
        
%% for first dataset
data1 = ones(100, 2);
data1 = data1 * 3 + randn(100, 2);

data2 = ones(100, 2);
data2 = data2 + randn(100, 2);
input = [data1; data2];

labels = [ones(100, 1); ones(100, 1) * (-1)];  
X = input;
Y = labels;

sigma = 1;
% for lineaer
K = zeros(n, n);
for i = 1:n
    for j = 1:n
        K(i, j) = exp(-1 * sum((X(i, :) - X(j, :)).^2) / (2*(sigma^2)));
    end
end

cvx_begin
    variable alphatrain(n);
    maximize ((ones(n, 1)'* alphatrain) - 0.5.*quad_form(Y.*alphatrain, K))
    subject to
        alphatrain>=0
        alphatrain<=C
        sum(Y.*alphatrain)==0
cvx_end

alphatrain(alphatrain < 1e-5) = 0;
Xs = X(alphatrain > 0, :);
Ys = Y(alphatrain > 0, :);
alphatrain = alphatrain(alphatrain > 0);

ns = size(alphatrain, 1);
K2 = zeros(n, ns);
for i = 1:n
    for j = 1:ns
        K2(i, j) = exp(-1 * sum((X(i, :) - Xs(j, :)).^2) / (2*(sigma^2)));
    end
end
b = mean(Y - K2*(alphatrain.*Ys));

% predict of training set
Ks = zeros(n, ns);
for i = 1:n
    for j = 1:ns
        Ks(i, j) = exp(-1 * sum((X(i, :) - Xs(j, :)).^2) / (2*(sigma^2)));
    end
end
scores = Ks*(alphatrain.*Ys) - b;

% predict of grids
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

len = size(xGrid, 1);
Ks = zeros(len, ns);
for i = 1:len
    for j = 1:ns
        Ks(i, j) = exp(-1 * sum((xGrid(i, :) - Xs(j, :)).^2) / (2*(sigma^2)));
    end
end

scores = Ks*(alphatrain.*Ys) + b;

figure;
h(1:2) = gscatter(X(:,1), X(:,2), labels,'rb','.');
hold on
contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),[0 0],'k');
contour(x1Grid,x2Grid,reshape(scores-1,size(x1Grid)),[0 0],'b');
contour(x1Grid,x2Grid,reshape(scores+1,size(x1Grid)),[0 0],'r');
title('hard margin SVM')
axis equal
hold off  

%% ploy 3
X = input;
Y = labels;
sigma = 1;
% for lineaer
K = zeros(n, n);
for i = 1:n
    for j = 1:n
        K(i, j) = (X(i, :) * X(j, :)')^3;
    end
end

cvx_begin
    variable alphatrain(n);
    maximize ((ones(n, 1)'* alphatrain) - 0.5.*quad_form(Y.*alphatrain, K))
    subject to
        alphatrain>=0
        alphatrain<=C
        sum(Y.*alphatrain)==0
cvx_end

alphatrain(alphatrain < 1e-5) = 0;
Xs = X(alphatrain > 0, :);
Ys = Y(alphatrain > 0, :);
alphatrain = alphatrain(alphatrain > 0);

n = 200;
ns = size(alphatrain, 1);
K2 = zeros(n, ns);
for i = 1:n
    for j = 1:ns
        K2(i, j) = (X(i, :) * Xs(j, :)')^3;
    end
end
b = mean(Y - K2*(alphatrain.*Ys));

% predict of training set
Ks = zeros(n, ns);
for i = 1:n
    for j = 1:ns
        Ks(i, j) = (X(i, :) * Xs(j, :)')^3;
    end
end
scores = Ks*(alphatrain.*Ys) - b;

% predict of grids
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

len = size(xGrid, 1);
Ks = zeros(len, ns);
for i = 1:len
    for j = 1:ns
        Ks(i, j) = (xGrid(i, :) * Xs(j, :)')^3;
    end
end

scores = Ks*(alphatrain.*Ys) + b;

figure;
h(1:2) = gscatter(X(:,1), X(:,2), labels,'rb','.');
hold on
contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),[0 0],'k');
contour(x1Grid,x2Grid,reshape(scores-1,size(x1Grid)),[0 0],'b');
contour(x1Grid,x2Grid,reshape(scores+1,size(x1Grid)),[0 0],'r');
title('hard margin SVM')
axis equal
hold off
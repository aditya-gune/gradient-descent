%Sample code source:
%http://www.dsplog.com/2011/10/29/batch-gradient-descent/

filename = 'D:\Aditya\Desktop\test2 p1-16.csv';
x = xlsread(filename, 'A:A');
y = xlsread(filename, 'B:B');
[n,m] = size(x); %make dynamic once code works

theta_vec = ones(m,n);

alpha =0.002;
err = zeros(m,n);

for steps = 1:10
    disp(steps)
    h_theta = (x*theta_vec);
    h_theta_v = h_theta*ones(n,1);
	y_v = y*ones(1,n);
	theta_vec = theta_vec - alpha*1/m*sum((y_v - h_theta_v).*x).';
	%err(:,steps) = 1/m*sum((y_v - h_theta_v).*x).';
end

figure;
plot(x(:,2),y,'bs-');
hold on
plot(x(:,2),x*theta_vec,'rp-');
grid on;
xlabel('x input');
ylabel('y output');
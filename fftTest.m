N = 10000; %% number of points 
T = 3.4; %% define time of interval, 3.4 seconds
t = [0:N-1]/N; %% define time
t = t*T; %% define time in seconds
f = sin(2*pi*1*t); %% define function, 10 Hz sine wave

figure(1);
plot(t,f);
ylim([-2 2]);

figure(2);
pf = fft(f);
p = abs(fft(f))/(N/2); %% absolute value of the fft
p = p(1:N/2).^2; %% take the power of positve freq. half
freq = [0:N/2-1]/T; %% find the corresponding frequency in Hz
%semilogy(freq,p); %% plot on semilog scale
plot(freq,p);
axis([0 20 0 1]); %% zoom in

figure(3);
f2 = ifft(pf);
plot(t,f2);
ylim([-2 2]);

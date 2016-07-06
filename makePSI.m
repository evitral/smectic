clear all;

file1 = 'psi.dat';
file2 = 'L1.dat';
file3 = 'stress.dat';
file4 = 'velu.dat';
file5 = 'velv.dat';
file6 = 'lapV.dat';

stressPlug = 1;

zData = importdata(file1);

dt = 0.01;
np = 256;
Z = zeros(np,np);

for i = 1:1:np
    Z(i,1:np) = zData((1+np*(i-1)):np*i);
end

size = 32;
dx = size/(np);

[X,Y] = meshgrid(dx/2:dx:(size-dx/2),dx/2:dx:(size-dx/2));

figure(1);

surf(X,Y,Z);

xlim([0 size]);
ylim([0 size]);

shading interp;
axis equal; 
view(0,90);
xlabel('x');
ylabel('y    ','rot',0);

figure(2);

L1Data = importdata(file2);

L1len = length(L1Data);

plot(dt:10*dt:10*dt*L1len,L1Data);
set(gca,'Yscale','log');
xlabel('t');
ylabel('L1    ','rot',0);

if stressPlug == 1
    
    stressData = importdata(file3);

    stress = zeros(np,np);

    for i = 1:1:np
        stress(i,1:np) = stressData((1+np*(i-1)):np*i);
    end

    [X,Y] = meshgrid(dx/2:dx:(size-dx/2),dx/2:dx:(size-dx/2));

    figure(3);

    Dstress  = zeros(np,np);
    
    for j = 1:1:np
    for i = 2:1:(np-1)
        Dstress(i,j) = (1/(2*dx))*(stress(i+1,j)-stress(i-1,j));
    end
    end
    
    surf(X,Y,Dstress);

    xlim([0 size]);
    ylim([0 size]);

    shading interp;
    axis equal; 
    view(0,90);
    xlabel('x');
    ylabel('y    ','rot',0);
end

figure(4);

zuData = importdata(file4);
zvData = importdata(file5);
lapVData = importdata(file6);


for i = 1:1:np
    Zu(i,1:np) = zuData((1+np*(i-1)):np*i);
    Zv(i,1:np) = zvData((1+np*(i-1)):np*i);
    lapV(i,1:np) = lapVData((1+np*(i-1)):np*i);
end

surf(X,Y,Zu);

xlim([0 size]);
ylim([0 size]);

shading interp;
axis equal; 
view(0,90);
xlabel('x');
ylabel('y    ','rot',0);

figure(5);

surf(X,Y,Zv);

xlim([0 size]);
ylim([0 size]);

shading interp;
axis equal; 
view(0,90);
xlabel('x');
ylabel('y    ','rot',0);

figure(6);

surf(X,Y,lapV);

xlim([0 size]);
ylim([0 size]);

shading interp;
axis equal; 
view(0,90);
xlabel('x');
ylabel('y    ','rot',0);
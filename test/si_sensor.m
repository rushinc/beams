%% Set Layer Geometry

% Unit cell parameters
D_x = 810; % Length of one period in 'x'
D_x = 2*round(D_x/2); % Scaled and rounded to nearest even number
D_y = D_x; % Length of one period in 'y'
N_z = 1; % No. of layers for Fourier analysis
h_z = 320; % Thickness of layers
h_g = 125; % Thickness of layers
z_l = [0,cumsum(h_z(1:N_z-1))]; % 'z' coordinate of each layer
L = 600;
w_1 = 230; % Width of Si bar 1
w_2 = 190; % Width of Si bar 2
C_1 = [405,405 - (w_1 + h_g)/2]; % Center of Si bar 1
C_2 = [405,405 + (w_2 + h_g)/2]; % Center of Si bar 1
C_c = [D_x/2,D_y/2]; % Center of unit cell
N_l = N_z + 2; % Total layers

%% Generate layers

% Build the structure as a binary 3D matrix
zoom = 4;
G_x = zoom*D_x; G_y = zoom*D_y;
indexProfLog = false(G_x,G_y,N_z);
centre = [G_x,G_y]/2;
w_1z = zoom*w_1;
w_2z = zoom*w_2;
L_z = zoom*L;
C_1z = zoom*C_1;
C_2z = zoom*C_2;
C_cz = zoom*C_c;

for nn=1:N_z
%     indexProfLog(C_cz(1)-L_z/2:C_cz(1)+L_z/2,...
%         C_cz(2)-w_1z/2:C_cz(2)+w_1z/2,nn) = true;
    indexProfLog(C_1z(1)-L_z/2:C_1z(1)+L_z/2,...
        C_1z(2)-w_1z/2:C_1z(2)+w_1z/2,nn) = true;
    indexProfLog(C_2z(1)-L_z/2:C_2z(1)+L_z/2,...
        C_2z(2)-w_2z/2:C_2z(2)+w_2z/2,nn) = true;
end

%% Source

y = 1653.8; % Range of wavelength in nm

% Angles
th = 0; % Incidence Angle (deg)
phi = 0; % Angle between 'x' axis and plane of incidence (deg)
psi = 0; % Polarization angle (deg) 0 -> TM, 90 -> TE

%% Materials
n_Air = 1;
n_Sen = 1.25;
n_Si = 3.46;
n_SiO2 = 1.5;

%% Initialization

% Fourier modes to consider in calculations (ODD)
N_x = 9; %=1; for uncarved % Number of Fourier modes in x direction 
N_y = 9; %=1; for uncarved % Number of Fourier modes in y direction
Nt = N_x*N_y;

% Preallocations
ky = 2*pi./y; % Wavenumbers
I = eye(Nt); II = eye(2*Nt); % Identity matrices
ZZ = zeros(2*Nt); zz = zeros(2*Nt,1);
EPS = zeros(Nt,Nt,N_l);
EPSyx = zeros(Nt,Nt,N_z);
EPSxy = zeros(Nt,Nt,N_z);
W = zeros(2*Nt,2*Nt,N_l);
V = zeros(2*Nt,2*Nt,N_l);
X = zeros(2*Nt,2*Nt,N_l);
qm = zeros(2*Nt,N_l);

% Foruier mode numbers and arrays
mx = (-(N_x-1)/2:(N_x-1)/2)';
my = (-(N_y-1)/2:(N_y-1)/2);
m0x = (N_x+1)/2;
m0y = (N_y+1)/2;
m0 = (Nt+1)/2;
    
%% Structure Analysis
% Permittivity and refractive indices
n_1 = n_Air; % Refractive index of incidence medium
n_2 = n_Si; % Refractive index of material in structure
n_3 = n_SiO2; % Refractive index of transmission medium

% Permittivity matrices for each layer
indexProf = ones(G_x,G_y);
for nn=1:N_z
    indexProf(indexProfLog(:,:,nn)) = n_2^2;
    indexProf(~indexProfLog(:,:,nn)) = n_Sen^2;

    eps_fft = fftshift(fft2(indexProf))/(G_x*G_y);
    eps_mn = eps_fft(G_x/2+2-N_x:G_x/2+N_x,G_y/2+2-N_y:G_y/2+N_y);

    for pp = 1:N_x
        for qq = 1:N_y
            EE = rot90(eps_mn(pp:pp+N_x-1,qq:qq+N_y-1),2);
            EPS(pp+N_x*(qq-1),:,nn+1) = reshape(EE,1,[]);
        end
    end

    i_iepsx_mj = zeros(N_x,G_y,N_x);
    iepsx_fft = fftshift(fft(indexProf.^(-1),[],1),1)/(G_x);

    for qq=1:G_y
        iepsx_m = iepsx_fft(G_x/2+2-N_x:G_x/2+N_x,qq);
        iepsx_mj = toeplitz(iepsx_m(N_x:2*N_x-1),flip(iepsx_m(1:N_x)));

        i_iepsx_mj(:,qq,:) = inv(iepsx_mj);
    end

    epsxy_fft = fftshift(fft(i_iepsx_mj,[],2),2)/(G_y);
    epsxy_mnj = epsxy_fft(:,G_y/2+2-N_y:G_y/2+N_y,:);

    E4 = zeros(N_x,N_y,N_x,N_y);

    for pp = 1:N_x
        for qq = 1:N_x
            E4(pp,:,qq,:) = toeplitz(epsxy_mnj(pp,N_y:2*N_y-1,qq),flip(epsxy_mnj(pp,1:N_y,qq)));
        end
    end

    EPSxy(:,:,nn) = reshape(E4,[N_x*N_y,N_x*N_y]);

    i_iepsy_nl = zeros(G_x,N_y,N_y);
    iepsy_fft = fftshift(fft(indexProf.^(-1),[],2),2)/(G_y);

    for pp=1:G_x
        iepsy_n = iepsy_fft(pp,G_y/2+2-N_y:G_y/2+N_y);
        iepsy_nl = toeplitz(iepsy_n(N_y:2*N_y-1),flip(iepsy_n(1:N_y)));

        i_iepsy_nl(pp,:,:) = inv(iepsy_nl);
    end

    epsyx_fft = fftshift(fft(i_iepsy_nl,[],1),1)/(G_x);
    epsyx_mnl = epsyx_fft(G_x/2+2-N_x:G_x/2+N_x,:,:);

    E4 = zeros(N_x,N_y,N_x,N_y);

    for rr = 1:N_y
        for ss = 1:N_y
            E4(:,rr,:,ss) = toeplitz(epsyx_mnl(N_x:2*N_x-1,rr,ss),flip(epsyx_mnl(1:N_x,rr,ss)));
        end
    end

    EPSyx(:,:,nn) = reshape(E4,[N_x*N_y,N_x*N_y]);
end

%% Begin computation in wavelength, polarization and incidence angle
% Incident Field
u_x = cosd(psi)*cosd(phi)*cosd(th) - sind(psi)*sind(phi);
u_y = cosd(psi)*sind(phi)*cosd(th) + sind(psi)*cosd(phi);
inc = zz; inc(m0) = u_x; inc(Nt+m0) = u_y;

% Wavenumber Matrices
k_0 = ky;

k_xi = k_0*n_1*sind(th)*cosd(phi) + 2*pi*mx/D_x;
k_yi = k_0*n_1*sind(th)*sind(phi) + 2*pi*my/D_y;
k_x_mn = reshape(repmat(k_xi,1,N_y),[],1);
k_y_mn = reshape(repmat(k_yi,N_x,1),[],1);

k_1_zmn = conj(((n_1*k_0)^2*ones(Nt,1) - k_x_mn.^2 - k_y_mn.^2).^(1/2));
k_3_zmn = conj(((n_3*k_0)^2*ones(Nt,1) - k_x_mn.^2 - k_y_mn.^2).^(1/2));

K_x = 1i*diag(k_x_mn/k_0);
K_y = 1i*diag(k_y_mn/k_0);

%% Eigenmatrices

% Eigenvalue formulation for the grating region
for nn=1:N_z
    F11 = -K_x*(EPS(:,:,nn+1)\K_y);
    F12 = I + K_x*(EPS(:,:,nn+1)\K_x);
    F21 = -I - K_y*(EPS(:,:,nn+1)\K_y);
    F22 = K_y*(EPS(:,:,nn+1)\K_x);
    F = [F11 F12;F21 F22];

    G11 = -K_x*K_y;
    G12 = EPSyx(:,:,nn) + K_x^2;
    G21 = -EPSxy(:,:,nn) - K_y^2;
    G22 = K_x*K_y;
    G = [G11 G12;G21 G22];

    [W(:,:,nn+1), Q] = eig(F*G);
    Q = (sqrt(Q));
    qm(:,nn+1) = diag(Q);
    V(:,:,nn+1) = -G*W(:,:,nn+1)/Q;
    X(:,:,nn+1) = diag(exp(-k_0*qm(:,nn+1)*h_z(nn)));
end

% Eigenmatrices for reflection and transmission layers
W(:,:,1) = II;
W(:,:,N_l) = II;

EPS(:,:,1) = n_1^2*I;
VIxx = (1i/k_0)*diag((k_x_mn.*k_y_mn)./k_1_zmn);
VIxy = (1i/k_0)*diag((k_y_mn.^2 + k_1_zmn.^2)./k_1_zmn);
VIyx = -(1i/k_0)*diag((k_x_mn.^2 + k_1_zmn.^2)./k_1_zmn);
VIyy = -VIxx;
V(:,:,1) = [VIxx VIxy;VIyx VIyy];
qm(:,1) = 1i*[k_1_zmn;k_1_zmn]/k_0;
X(:,:,1) = II;

EPS(:,:,N_l) = n_3^2*I;
VIIxx = (1i/k_0)*diag((k_x_mn.*k_y_mn)./k_3_zmn);
VIIxy = (1i/k_0)*diag((k_y_mn.^2 + k_3_zmn.^2)./k_3_zmn);
VIIyx = -(1i/k_0)*diag((k_x_mn.^2 + k_3_zmn.^2)./k_3_zmn);
VIIyy = -VIIxx;
V(:,:,N_l) = [VIIxx VIIxy;VIIyx VIIyy];
qm(:,N_l) = 1i*[k_3_zmn;k_3_zmn]/k_0;
X(:,:,N_l) = II;

%% Big matrix calculation

% Preallocation
B = repmat(ZZ,2*(N_l+1),2*(N_l+1));

% Building B using b
for ll=1:N_l
    b = [W(:,:,ll) W(:,:,ll)*X(:,:,ll);
        V(:,:,ll) -V(:,:,ll)*X(:,:,ll);
        -W(:,:,ll)*X(:,:,ll) -W(:,:,ll);
        -V(:,:,ll)*X(:,:,ll) V(:,:,ll)];
    
    B(4*(ll-1)*Nt+1:4*(ll+1)*Nt,4*(ll-1)*Nt+1:4*ll*Nt) = b;
end

% The final two columns
b_l=[-W(:,:,1), ZZ;
    V(:,:,1), ZZ;
    repmat(ZZ,2*(N_l-1),2);
    ZZ, W(:,:,N_l);
    ZZ, V(:,:,N_l)];

B(:,4*N_l*Nt+1:4*(N_l+1)*Nt) = b_l; % The BIG matrix

% Incident field matrix
Ein = inc;
Hin = V(:,:,1)*Ein;
Fin = [Ein;Hin;repmat(zz,2*N_l,1)]; % Fields in each layer at incidence

C = B\Fin; % Column matrix with wave amplitudes for each layer

% Reflected and transmitted fourier modes
Cc = reshape(C,4*Nt,N_l+1); % Rearrange
Cc(2*Nt+1:4*Nt,N_l) = zz; % Forcing incidence from transmission layer to 0

R_xy = Cc(1:2*Nt,N_l+1);
T_xy = Cc(2*Nt+1:4*Nt,N_l+1);

%% Efficiency

R_z = (k_x_mn./k_1_zmn).*R_xy(1:Nt) + (k_y_mn./k_1_zmn).*R_xy(Nt+1:2*Nt);
T_z = -(k_x_mn./k_3_zmn).*T_xy(1:Nt) - (k_y_mn./k_3_zmn).*T_xy(Nt+1:2*Nt);

R_xyz = [R_xy;R_z]; T_xyz = [T_xy;T_z];

DEri = real([(k_1_zmn/k_1_zmn(m0));(k_1_zmn/k_1_zmn(m0));(k_1_zmn/k_1_zmn(m0))]).*(R_xyz.*conj(R_xyz));
DEti = real([(k_3_zmn/k_1_zmn(m0));(k_3_zmn/k_1_zmn(m0));(k_3_zmn/k_1_zmn(m0))]).*(T_xyz.*conj(T_xyz));

DEr = sum(DEri(:));
DEt = sum(DEti(:));

%% Fields

% Space parameters
rz = y/10; % Resolution in 'z'
zStart = -20; % Start plotting from this z coordinate
zEnd = sum(h_z) - zStart; % Stop plotting at this z coordinate
z = h_z/2; % zStart:rz:zEnd;

rx = 5; % Resolution in 'x'
x = 1:rx:D_x-rx+1; % Need to compute just a single period

ry = 5; % Resolution in 'y'
y = 1:rx:D_y-rx+1;

zi = 1; % Starting z index
ll = 1; % Starting layer

% z coordinate for the end of each layer
z_l = cumsum([0,h_z,Inf]);

% Preallocating field magnitude matrices
eps_z = zeros(length(x),length(y),length(z));
E_x = zeros(length(x),length(y),length(z));
E_y = zeros(length(x),length(y),length(z));
E_z = zeros(length(x),length(y),length(z));
H_x = zeros(length(x),length(y),length(z));
H_y = zeros(length(x),length(y),length(z));
H_z = zeros(length(x),length(y),length(z));
E_v = zeros(length(x),length(y),length(z),3);
H_v = zeros(length(x),length(y),length(z),3);
S_v = zeros(length(x),length(y),length(z),3);

while zi<=length(z)
    % Propagation in z
    if z(zi)<z_l(ll)
        p_z = exp(-k_0*qm(:,ll)*(z(zi)-z_l(ll-(ll~=1))));
        Phi_z = diag([p_z;X(:,:,ll)*p_z.^(-1)]);
        
        M = [W(:,:,ll), W(:,:,ll);
             EPS(:,:,ll)\[-K_y,K_x]*V(:,:,ll), -EPS(:,:,ll)\[-K_y,K_x]*V(:,:,ll);
             V(:,:,ll), -V(:,:,ll);
             [-K_y,K_x]*W(:,:,ll), [-K_y,K_x]*W(:,:,ll)];
             
        Psi_z = M*Phi_z*Cc(:,ll); % Field amplitude coefficients at (*,*,z)
        
        for yi=1:length(y)
            p_y = exp(-k_0*diag(K_y)*y(yi)); % Propagation in x
            Phi_y = diag(repmat(p_y,6,1));
    
            Psi_yz = Phi_y*Psi_z; % Field amplitude coefficients at (*,y,z)
            for xi=1:length(x)
                p_x = exp(-k_0*diag(K_x)*x(xi)); % Propagation in x
                Phi_x = diag(repmat(p_x,6,1));
    
                Psi_xyz = Phi_x*Psi_yz; % Field amplitude coefficients at (x,y,z)
    
                % Separating the field components from \Psi(x,z)
                E_x(xi,yi,zi) = sum(Psi_xyz(1:Nt));
                E_y(xi,yi,zi) = sum(Psi_xyz(Nt+1:2*Nt));
                E_z(xi,yi,zi) = sum(Psi_xyz(2*Nt+1:3*Nt));
                H_x(xi,yi,zi) = sum(Psi_xyz(3*Nt+1:4*Nt));
                H_y(xi,yi,zi) = sum(Psi_xyz(4*Nt+1:5*Nt));
                H_z(xi,yi,zi) = sum(Psi_xyz(5*Nt+1:6*Nt));
                
                E_v(xi,yi,zi,:) = [E_x(xi,yi,zi),E_y(xi,yi,zi),E_z(xi,yi,zi)];
                H_v(xi,yi,zi,:) = [H_x(xi,yi,zi),H_y(xi,yi,zi),H_z(xi,yi,zi)];
                S_v(xi,yi,zi,:) = 0.5*real(cross(E_v(xi,yi,zi,:),conj(H_v(xi,yi,zi,:))));
            end
        end
        zi = zi + 1;
    else
        ll = ll + 1;
    end
end

%% Plot

absE2 = []; absH2 = [];
absE2(:,:) = abs(E_x(:,:,1)).^2 + abs(E_y(:,:,1)).^2 + abs(E_z(:,:,1)).^2;
absH2(:,:) = abs(H_x(:,:,1)).^2 + abs(H_y(:,:,1)).^2 + abs(H_z(:,:,1)).^2;

zz = 1;
figure('Units','inches','Position',[5 2 7 5]);
h = pcolor(x,y,absH2);
set(h,'linestyle','none');
set(gca,'fontsize',20);
set(gca, 'FontName', 'Times New Roman');
colormap('jet');
bb = colorbar;
bb.Label.String = '|H|^2';
bb.Label.Rotation = 270;
clims = get(gca,'clim');
bb.Label.Position = [4,(clims(1)+clims(2))/2,0];
title('$\lambda=1550$ nm','interpreter','latex')
xlabel('$x$ (nm)','interpreter','latex');
ylabel('$y$ (nm)','interpreter','latex');
hold on
quiver(x(1:10:end),y(1:10:end),imag(H_x(1:10:end,1:10:end,zz)),imag(H_y(1:10:end,1:10:end,zz)))
axis equal
xlim([min(x),max(x)])
ylim([min(y),max(y)])

figure('Units','inches','Position',[5 2 7 5]);
h = pcolor(x,y,absE2);
set(h,'linestyle','none');
set(gca,'fontsize',20);
set(gca, 'FontName', 'Times New Roman');
colormap('jet');
bb = colorbar;
bb.Label.String = '|E|^2';
bb.Label.Rotation = 270;
clims = get(gca,'clim');
bb.Label.Position = [4,(clims(1)+clims(2))/2,0];
title('$\lambda=1550$ nm','interpreter','latex')
xlabel('$x$ (nm)','interpreter','latex');
ylabel('$y$ (nm)','interpreter','latex');
hold on
quiver(x(1:10:end),y(1:10:end),real(E_x(1:10:end,1:10:end,zz)),real(E_y(1:10:end,1:10:end,zz)))
axis equal
xlim([min(x),max(x)])
ylim([min(y),max(y)])

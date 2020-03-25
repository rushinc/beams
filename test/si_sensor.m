%% Inputs and Structure Parameters

y = 1500; % Range of wavelength in nm

% Angles
th = 0; % Incidence Angle (deg)
phi = 0; % Angle between 'x' axis and plane of incidence (deg)
psi = 0; % Polarization angle (deg) 0 -> TM, 90 -> TE

n_sweep = 1;
R_ww = zeros(length(n_sweep),length(y));

for ww=1:length(n_sweep)
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

% Fourier modes to consider in calculations (ODD)
N_x = 9; %=1; for uncarved % Number of Fourier modes in x direction 
N_y = 9; %=1; for uncarved % Number of Fourier modes in y direction
Nt = N_x*N_y;

% Preallocations
ky = 2*pi./y; % Wavenumbers
I = eye(Nt); II = eye(2*Nt); % Identity matrices
ZZ = zeros(2*Nt);
EPS = zeros(Nt,Nt,N_z);
EPSyx = zeros(Nt,Nt,N_z);
EPSxy = zeros(Nt,Nt,N_z);
W = zeros(2*Nt,2*Nt,N_l);
V = zeros(2*Nt,2*Nt,N_l);
X = zeros(2*Nt,2*Nt,N_l);
DEr = zeros(length(y),length(th),length(psi));
DEt = zeros(length(y),length(th),length(psi));

% Foruier mode numbers and arrays
mx = (-(N_x-1)/2:(N_x-1)/2)';
my = (-(N_y-1)/2:(N_y-1)/2);
m0x = (N_x+1)/2;
m0y = (N_y+1)/2;
m0 = (Nt+1)/2;

%% Materials
n_Air = 1;
n_Sen = n_sweep(ww);
n_Si = 3.46;
n_SiO2 = 1.5;

%% Index profile of moth eye structures
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
    
%% Structure Analysis

% Permittivity and refractive indices
n_1 = n_Air; % Refractive index of incidence medium
n_2 = n_Si; % Refractive index of material in structure
n_3 = n_SiO2; % Refractive index of transmission medium

% Permittivity matrices for each layer
indexProf = ones(G_x,G_y);
for nn=1:N_z
    indexProf(indexProfLog(:,:,nn)) = n_2^2;
    if nn==2
        indexProf(~indexProfLog(:,:,nn)) = n_Air^2;
    else
        indexProf(~indexProfLog(:,:,nn)) = n_Sen^2;
    end

    eps_fft = fftshift(fft2(indexProf))/(G_x*G_y);
    eps_mn = eps_fft(G_x/2+2-N_x:G_x/2+N_x,G_y/2+2-N_y:G_y/2+N_y);

    for pp = 1:N_x
        for qq = 1:N_y
            EE = rot90(eps_mn(pp:pp+N_x-1,qq:qq+N_y-1),2);
            EPS(pp+N_x*(qq-1),:,nn) = reshape(EE,1,[]);
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

%% Begin loop in wavelength, polarization and incidence angle

for yy=1:length(y)
    uy = y(yy)/1000; % Wavelength in micron
    for ss=1:length(psi)
        for tt=1:length(th)
            % Incident Field
            u_x = cosd(psi(ss))*cosd(phi)*cosd(th(tt)) - sind(psi(ss))*sind(phi);
            u_y = cosd(psi(ss))*sind(phi)*cosd(th(tt)) + sind(psi(ss))*cosd(phi);

            % Wavenumber Matrices
            k_0 = ky(yy);

            k_xi = k_0*n_1*sind(th(tt))*cosd(phi) + 2*pi*mx/D_x;
            k_yi = k_0*n_1*sind(th(tt))*sind(phi) + 2*pi*my/D_y;
            k_x_mn = reshape(repmat(k_xi,1,N_y),[],1);
            k_y_mn = reshape(repmat(k_yi,N_x,1),[],1);

            k_1_zmn = conj(((n_1*k_0)^2*ones(Nt,1) - k_x_mn.^2 - k_y_mn.^2).^(1/2));
% 			k_2_zmn = conj(((n_2*k_0)^2*ones(Nt,1) - k_x_mn.^2 - k_y_mn.^2).^(1/2));
            k_3_zmn = conj(((n_3*k_0)^2*ones(Nt,1) - k_x_mn.^2 - k_y_mn.^2).^(1/2));

            K_x = 1i*diag(k_x_mn/k_0);
            K_y = 1i*diag(k_y_mn/k_0);

            %% Eigenmatrices

            % Eigenvalue formulation for the grating region
            for nn=1:N_z
                F11 = -K_x*(EPS(:,:,nn)\K_y);
                F12 = I + K_x*(EPS(:,:,nn)\K_x);
                F21 = -I - K_y*(EPS(:,:,nn)\K_y);
                F22 = K_y*(EPS(:,:,nn)\K_x);
                F = [F11 F12;F21 F22];

                G11 = -K_x*K_y;
                G12 = EPSyx(:,:,nn) + K_x^2;
                G21 = -EPSxy(:,:,nn) - K_y^2;
                G22 = K_x*K_y;
                G = [G11 G12;G21 G22];

                [W(:,:,nn+1), Q] = eig(F*G);
                Q = (sqrt(Q));
                qm = diag(Q);
                V(:,:,nn+1) = -G*W(:,:,nn+1)/Q;
                X(:,:,nn+1) = diag(exp(-k_0*qm*h_z(nn)));
            end

            % Eigenmatrices for reflection and transmission layers
            W(:,:,1) = II;
            W(:,:,N_l) = II;

            VIxx = (1i/k_0)*diag((k_x_mn.*k_y_mn)./k_1_zmn);
            VIxy = (1i/k_0)*diag((k_y_mn.^2 + k_1_zmn.^2)./k_1_zmn);
            VIyx = -(1i/k_0)*diag((k_x_mn.^2 + k_1_zmn.^2)./k_1_zmn);
            VIyy = -VIxx;
            V(:,:,1) = [VIxx VIxy;VIyx VIyy];
            X(:,:,1) = II;

            VIIxx = (1i/k_0)*diag((k_x_mn.*k_y_mn)./k_3_zmn);
            VIIxy = (1i/k_0)*diag((k_y_mn.^2 + k_3_zmn.^2)./k_3_zmn);
            VIIyx = -(1i/k_0)*diag((k_x_mn.^2 + k_3_zmn.^2)./k_3_zmn);
            VIIyy = -VIIxx;
            V(:,:,N_l) = [VIIxx VIIxy;VIIyx VIIyy];
            X(:,:,N_l) = II;

            %% S matrix calculation

            % Preallocating
            Suu = II;
            Sud = ZZ;
            Sdu = ZZ;
            Sdd = II;

            for ll=1:N_l-1
                S = [W(:,:,ll+1) -W(:,:,ll);V(:,:,ll+1) V(:,:,ll)]\[W(:,:,ll)*X(:,:,ll) -W(:,:,ll+1)*X(:,:,ll+1);V(:,:,ll)*X(:,:,ll) V(:,:,ll+1)*X(:,:,ll+1)];

                % Cascade the next interface to the current S matrix
                Auu = Suu;
                Aud = Sud;
                Adu = Sdu;
                Add = Sdd;
                Buu = S(1:2*Nt,1:2*Nt);
                Bud = S(1:2*Nt,2*Nt+1:4*Nt);
                Bdu = S(2*Nt+1:4*Nt,1:2*Nt);
                Bdd = S(2*Nt+1:4*Nt,2*Nt+1:4*Nt);

                Suu = Buu*((II-Aud*Bdu)\Auu);
                Sud = Bud+(Buu*Aud*((II-Bdu*Aud)\Bdd));
                Sdu = Adu+(Add*Bdu*((II-Aud*Bdu)\Auu));
                Sdd = Add*((II-Bdu*Aud)\Bdd);
            end

            %% Efficiency

            R_xy = u_x*Sdu(:,m0)+u_y*Sdu(:,Nt+m0);
            T_xy = u_x*Suu(:,m0)+u_y*Suu(:,Nt+m0);

            R_z = (k_x_mn./k_1_zmn).*R_xy(1:Nt) + (k_y_mn./k_1_zmn).*R_xy(Nt+1:2*Nt);
            T_z = -(k_x_mn./k_3_zmn).*T_xy(1:Nt) - (k_y_mn./k_3_zmn).*T_xy(Nt+1:2*Nt);

            R_xyz = [R_xy;R_z]; T_xyz = [T_xy;T_z];

            DEri = real([(k_1_zmn/k_1_zmn(m0));(k_1_zmn/k_1_zmn(m0));(k_1_zmn/k_1_zmn(m0))]).*(R_xyz.*conj(R_xyz));
            DEti = real([(k_3_zmn/k_1_zmn(m0));(k_3_zmn/k_1_zmn(m0));(k_3_zmn/k_1_zmn(m0))]).*(T_xyz.*conj(T_xyz));

            DEr(yy,tt,ss) = sum(DEri(:));
            DEt(yy,tt,ss) = sum(DEti(:));
            disp([num2str(yy),'/',num2str(length(y)),', ',num2str(ww),'/',num2str(length(n_sweep))]) % Display progress
        end
    end
end

%% Plot

% Angular Spectrum
pp = 1; % Select the polarization for the plot in terms of psi(pp)
% figure('Units','inches','Position',[5 2 6 5]);
% h = pcolor(y/1000,th,DEr(:,:,pp)');
% set(h,'linestyle','none');
% set(gca,'fontsize',20);
% colormap('jet');
% bb = colorbar;
% bb.Label.String = 'Reflectance';
% bb.Label.Rotation = 270;
% clims = get(gca,'clim');
% bb.Label.Position = [3.75,(clims(1)+clims(2))/2,0];
% caxis([0,1]);
% xlabel('$\lambda (\mum)$','interpreter','latex');
% ylabel('$\theta (^o)$','interpreter','latex');

% Line Plot
tt = 1; % [1,121,241]; % Incidence angles to be plotted in terms of th(tt)
% figure('Units','inches','Position',[5 2 6 5]);
hold on
plot(y/1000,DEr(:,tt(1),pp),'linewidth',2);
% hold on
for aa=2:length(tt)
    plot(y/1000,DEr(:,tt(aa),pp),'linewidth',2);
end
xlim([min(y),max(y)]/1000);
xlabel('\lambda (μm)');
ylabel('Reflectance','interpreter','latex');
% title(['$\phi = ',num2str(phi),'^\circ$'],'interpreter','latex');
set(gca,'fontsize',18);
set(gca,'fontname','Times New Roman');
ylim([0,1]);
labels = cell(size(tt));
for aa=1:length(tt)
    labels(aa) = {['\theta = ',num2str(th(tt(aa))),'\circ']};
end

R_ww(ww,:) = DEr(:,tt(1),pp);

end

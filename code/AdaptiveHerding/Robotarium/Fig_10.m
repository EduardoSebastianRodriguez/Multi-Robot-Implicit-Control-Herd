% ROBOTARIUM EXPERIMENT
% --------------------------------------------------------------------------------
% 3 herders vs 2 evaders
%
% Evaders behave following: Inverse Model (1) and Exponential Model (1)
% --------------------------------------------------------------------------------
% Please, if you find this file useful, consider citing us: 
% --------------------------------------------------------------------------------
% E. Sebastián, E. Montijano and C. Sagüés, 
% “Adaptive Multirobot Implicit Control of Heterogeneous Herds,” 
% IEEE Transactions on Robotics, vol. 38, no. 6, pp. 3622-3635, 2022. 
% --------------------------------------------------------------------------------
% [More info at]: https://sites.google.com/unizar.es/poc-team/research/mrherding
% [Video]:        https://www.youtube.com/watch?v=U5KjP-2H1BM
% [Arxiv]:        https://arxiv.org/abs/2206.05888
% --------------------------------------------------------------------------------
% Eduardo Sebastián -- https://eduardosebastianrodriguez.github.io/
%
% Ph.D. Candidate
% Departamento de Informática e Ingeniería de Sistemas
% Universidad de Zaragoza
% --------------------------------------------------------------------------------
% Last modification: December 22, 2022
% --------------------------------------------------------------------------------
% [WARNING]: Robotarium-based codes may sometimes fail due to low-level
% stochastic processes in the sensing, communication and/or actuation
% protocols. 
% --------------------------------------------------------------------------------

close all
clear all
clc

%% Initialize stuff

% Define number of evaders and herders
N_herders = 3;
M_evaders = 2;

% Define initial desired position for the evaders and the vector for the time
% varying pattern 
x_d   = [0.00 0.3 -0.7 -0.3]';
x_b   = x_d;
v_d_x = [0.001 0.0025];
w_d_y = [0.0 0.03];
x_d_d = 0*x_d;

% Define initial position of the evaders 
x      = [-0.05 0.2 -0.55 -0.2]';
x_plot = x;
x_prev = x;

% Define initial position of the herders 
u      = [-1.4 0.0 0.0 -0.8 1.2 0.8]';
u_plot = u;
udot   = u;

% Define velocities
xdot   = zeros(2*M_evaders,1);
dposes = zeros(2,M_evaders+N_herders);
u_inici= zeros(2,N_herders);
u_final= zeros(2,N_herders);
dx     = zeros(2,M_evaders);
du     = zeros(2,N_herders);

% Approaching
x_avg = zeros(2,1);
x_std = zeros(2,2);
for i=1:M_evaders
    x_avg = x_avg + x(2*i-1:2*i)./M_evaders;
end
for i=1:M_evaders
    x_std(1,1) = x_std(1,1) + (x(2*i-1)-x_avg(1))*(x(2*i-1)-x_avg(1))./M_evaders;
    x_std(1,2) = x_std(1,2) + (x(2*i-1)-x_avg(1))*(x(2*i)-x_avg(2))./M_evaders;
    x_std(2,1) = x_std(2,1) + (x(2*i)-x_avg(2))*(x(2*i-1)-x_avg(1))./M_evaders;
    x_std(2,2) = x_std(2,2) + (x(2*i)-x_avg(2))*(x(2*i)-x_avg(2))./M_evaders;
end
radius = 0.25;
% Obtain SVD decomposition 
[U, D, Vgoal] = svd(x_std);

% Get "eigenaxis" 
agoal = 1/sqrt(D(1,1)) * radius;
if D(2) < 1e-10
    bgoal = agoal;
else
    bgoal = 1/sqrt(D(2,2)) * radius;
end

% Numeric parameters
T        = 0.033; % From Robotarium documentation
end_time = 0.0;

% Select the number of iterations for the experiment and stopping threshold
iterations        = 15000;
initialIterations = 500;

% Define space of the arena 
X_max = 1.6-0.1; % From Robotarium documentation
X_min = -1.6+0.1;% From Robotarium documentation
Y_max = 1.0-0.1; % From Robotarium documentation
Y_min = -1.0+0.1; % From Robotarium documentation

% Define parameters
theta_r = [0.02 0.05]';
theta_e = [0.015 0.04]';
betaS   = 0.5;
tau     = 1.0;
sigmaS  = 1.2;
angry   = false;
landa   = 0.0;
v_max   = 0.2;

% Define control gains and adaptive gains
K        = 0.1*eye(2*M_evaders);
Kstar    = 50*eye(2*M_evaders);
Kparams  = 200*eye(2*M_evaders);
variable = -0.0005;

% Define storing variables 
P   = zeros(2*M_evaders,iterations);
PD  = zeros(2*M_evaders,iterations);
H   = zeros(2*N_herders,iterations);
XDOT= zeros(2*M_evaders,iterations);
UDOT= zeros(2*N_herders,iterations);
C_r = zeros(M_evaders,iterations);
C_e = zeros(M_evaders,iterations);

%% Init robotarium
initial_conditions = [];
r = Robotarium('NumberOfRobots', M_evaders+N_herders, 'ShowFigure', true, 'InitialConditions', initial_conditions);

%% Go to initial configuration/pose

% Let's retrieve some of the tools we'll need.  We would like a
% single-integrator position controller, a single-integrator barrier
% function, and a mapping from single-integrator to unicycle dynamics
unicycle_barrier_certificate = create_uni_barrier_certificate('SafetyRadius', 0.13,'BarrierGain', 30);
      
%Get randomized initial conditions in the robotarium arena
final_goal_points = vertcat(reshape(horzcat(x',u'),[2,5]),[-2.3 0.7 0.0 1.6 -2.3]);

args = {'PositionError', 0.05, 'RotationError', 0.05};
init_checker = create_is_initialized(args{:});
controller = create_waypoint_controller(args{:});
controller_si = create_si_position_controller('XVelocityGain', 0.8, 'YVelocityGain', 0.8);

% Get initial location data for while loop condition.
poses=r.get_poses();
r.step();

while(~init_checker(poses, final_goal_points))
    poses = r.get_poses();
    dxu = controller(poses, final_goal_points);
    dxu = unicycle_barrier_certificate(dxu, poses);      

    r.set_velocities(1:M_evaders+N_herders, dxu);
    r.step();   
end

% We can call this function to debug our experiment!  Fix all the errors
% before submitting to maximize the chance that your experiment runs
% successfully.
r.debug();

%% Experiment begins

% Let's retrieve some of the tools we'll need.  We would like a
% single-integrator position controller, a single-integrator barrier
% function, and a mapping from single-integrator to unicycle dynamics
si_barrier_certificate = create_si_barrier_certificate('SafetyRadius', 0.13,'BarrierGain', 30);
si_to_uni_dynamics = create_si_to_uni_dynamics();

%Set desired position for evaders in the robotarium arena
poses_goal   = reshape(x_d,[2,2]);               
control_mode = zeros(N_herders,1);

% Iterate for the previously specified number of iterations
for k = 1:iterations
          
    % Record variables
    P(:,k)  = x;
    PD(:,k) = x_d;
    H(:,k)  = u;
    C_r(:,k)= theta_r;
    C_e(:,k)= theta_e;
    
    % Change in x_d
    if k>initialIterations && k<iterations
        for j=1:M_evaders
            x_d(2*j-1)   =  x_d(2*j-1) + v_d_x(j)*T; 
            x_d(2*j)     =  x_b(2*j)   + 0.07*sin(w_d_y(j)*k*T + 2*pi/j);
            x_d_d(2*j-1) =  v_d_x(j);
            x_d_d(2*j)   =  0.07*w_d_y(j)*cos(w_d_y(j)*k*T + 2*pi/j);
        end
    end   
        
    % Retrieve the most recent poses from the Robotarium.  The time delay is
    % approximately 0.033 seconds
    poses = r.get_poses();

    %% Proposed algorithm

    % Rebuild vectors to use the algorithm
    for i=1:M_evaders
        x(2*i-1) = poses(1,i);
        x(2*i)   = poses(2,i);
    end

    for i=1:N_herders
        u(2*i-1) = poses(1,i+M_evaders);
        u(2*i)   = poses(2,i+M_evaders);
    end

    % Store previous values for input and state
    u_prev = u;
    x_prev = x;
    
    % Update the state according to evaders' models
    xdot = 0*xdot;
    for j=1:1 % Inverse evader
        for i=1:N_herders
            xdot(2*j-1:2*j) = xdot(2*j-1:2*j) + theta_r(j)*( (x(2*j-1:2*j)-u(2*i-1:2*i))/norm(x(2*j-1:2*j)-u(2*i-1:2*i))^3 );
        end
        if norm(xdot(2*j-1:2*j)) > v_max
            xdot(2*j-1:2*j) = v_max*xdot(2*j-1:2*j)/norm(xdot(2*j-1:2*j));
        end
    end
    for j=2:2 % Exponential evader
        angry = false;
        var   = betaS*theta_r(j);
        for i=1:N_herders
            if norm(x(2*j-1:2*j)-u(2*i-1:2*i))<tau
                angry = true;
                var   = theta_r(j);
                break;
            end
        end
        for i=1:N_herders
            X = norm(x(2*j-1:2*j)-u(2*i-1:2*i))^2;
            xdot(2*j-1:2*j) = xdot(2*j-1:2*j) + var*(x(2*j-1:2*j)-u(2*i-1:2*i))*exp(-(1/(sigmaS^2))*X);
        end
        if norm(xdot(2*j-1:2*j)) > v_max
            xdot(2*j-1:2*j) = v_max*xdot(2*j-1:2*j)/norm(xdot(2*j-1:2*j));
        end
    end
    
    if k == 70
        control_mode = ones(N_herders,1);
    end
    
    if control_mode(1) == 0
        for iii=1:N_herders
            % Approach near the preys
            thetas = zeros(N_herders,1);
            for ii=1:N_herders
                thetas(ii) = mod(atan2(u(2*ii),u(2*ii-1)),2*pi);
            end
            theta_min = 0;
            theta_max = 0;
            theta_cen = 0;
            [sorted,order] = sort(thetas); 
            for ii=1:N_herders
                if order(ii) == iii
                    theta_cen = thetas(order(ii));
                    if ii+1 > N_herders
                        theta_max = thetas(order(1));
                    else
                        theta_max = thetas(order(ii+1));
                    end
                    if ii-1 < 1
                        theta_min = thetas(order(N_herders));
                    else
                        theta_min = thetas(order(ii-1));
                    end
                end
            end
            d1 = theta_min - theta_max;
            d2 = 2*pi - abs(d1);
            if d1 > 0
                d2 = -d2;
            end
            if abs(d1) < abs(d2)
                error = abs(d1);
                if theta_max > theta_min
                    sense = 1;
                else
                    sense = -1;
                end
            else
                error = abs(d2);
                if theta_max > theta_min
                    sense = -1;
                else
                    sense = 1;
                end
            end
            error = sense*error;
            if error < 0
                error = error + 2*pi;
            end
            
            theta_goal = theta_min + error/2;
            
            d1 = theta_cen - theta_goal;
            d2 = 2*pi - abs(d1);
            if d1 > 0
                d2 = -d2;
            end
            if abs(d1) < abs(d2)
                error = abs(d1);
                if theta_goal > theta_cen
                    sense = 1;
                else
                    sense = -1;
                end
            else
                error = abs(d2);
                if theta_goal > theta_cen
                    sense = -1;
                else
                    sense = 1;
                end
            end
            theta_dot = sense*error;
            theta_cen = theta_cen + T*theta_dot;
            theta_cen = mod(theta_cen,2*pi);
            state     = Vgoal*[agoal*cos(theta_cen);bgoal*sin(theta_cen)];
            
            diameterD    = norm(state);
            diameter     = norm(u(2*iii-1:2*iii));
            diameter_dot = diameterD - diameter;
            diameter     = diameter + T*diameter_dot;
            
            herder_goal = [diameter*cos(theta_cen),diameter*sin(theta_cen)];
            speed       = (herder_goal'-u(2*iii-1:2*iii))/T;
            
            if norm(speed) > v_max
                speed = v_max*speed/norm(speed);
            end

            % Apply CBF to surround the group of preys
            % Instantiate optimization variables 
            uOpt = optimvar("uOpt",2,1);
            
            % Problem
            prob = optimproblem('ObjectiveSense','minimize');
            
            % Objective function
            prob.Objective = (uOpt-speed)'*(uOpt-speed);

            % Constraints
            radius         = 0.025;
            
            % Obtain SVD decomposition 
            [U, D, V] = svd(x_std);

            % Get "eigenaxis" 
            a = 1/sqrt(D(1,1)) * radius;
            if D(2,2) < 1e-10
                b = a;
            else
                b = 1/sqrt(D(2,2)) * radius;
            end

            % Coordinate transform 
            point = V*u(2*iii-1:2*iii);
            try
                theta = fzero(@(theta) myfun(theta,a,b,point(1),point(2)),atan2(a*point(2),b*point(1)));
            catch
                disp('fail');
            end
            point = V*[a*cos(theta);b*sin(theta)];

            % Lfh(x)+Lgh(x)u+gamma*h**3(x)>=0 
            rel_pos        = point - u(2*iii-1:2*iii);
            h_cbf          = norm(rel_pos) + T*((reshape(rel_pos,[1,2])*reshape(speed,[2,1])))/norm(rel_pos) - 1;
            alpha_function = 50*(h_cbf^3);
            Lf             = -(point(1)-u(2*iii-1))/norm(point - u(2*iii-1:2*iii))*u(2*iii-1) - ...
                             (point(2)-u(2*iii))/norm(point - u(2*iii-1:2*iii))*u(2*iii);
            Lg             = -(point(1)-u(2*iii-1))/norm(point - u(2*iii-1:2*iii))*T*uOpt(1) - ...
                             (point(2)-u(2*iii))/norm(point - u(2*iii-1:2*iii))*T*uOpt(2);
            prob.Constraints.cons1  = Lf + Lg + alpha_function>=0;

            % Solve it
            options = optimoptions('quadprog','Display','none');
            sol     = solve(prob,'options',options);
            
            % Get input
            udot(2*iii-1:2*iii) = reshape(sol.uOpt,[2,1]);
            if norm(udot(2*iii-1:2*iii)) > v_max
                udot(2*iii-1:2*iii) = v_max*udot(2*iii-1:2*iii)/norm(udot(2*iii-1:2*iii));
            end
            tDot = zeros(M_evaders,1);  
        end
    else
        % Calculate estimated f(x,u) 
        [A,B] = buildSystem(x,u,theta_e,betaS,tau,sigmaS);
        f     = calculateF(x,u,A,B);

        % Calculate h(x,u)
        h = calculateH2(x,x_d,f,K,x_d_d);

        % Calculate h*
        hD = -Kstar*h;

        % Calculate theta*
        tD = buildTD(xdot,f,Kparams);

        % Calculate Jacobians 
        Jx = buildJx2(x,x_d,u,theta_e,betaS,tau,sigmaS,K,x_d_d);
        Ju = buildJu2(x,x_d,u,theta_e,betaS,tau,sigmaS,K,x_d_d);
        Jt = buildJparams(x,u,betaS,tau,sigmaS);

        % Calculate udot: udot = pseudoinverse(Ju)*(-Kstar*h - Jx*f)
        % 0.001*I is to ensure invertibility 
        uDot = Ju'/(Ju*Ju'+0.001*eye(2*M_evaders))*(hD - Jx*f);

        % Calculate params_dot: params_dot = pseudoinverse(J)*(-Kstar*h - Jx*f)
        % 0.001*I is to ensure invertibility 
        tDot = Jt'/(Jt*Jt' + 0.001*eye(2*M_evaders))*(tD - Jx*xdot - Ju*uDot);

        % The constant mitigates noises in the adaptation
        theta_e = theta_e + T*tDot*variable;

        
        for iii=1:N_herders
            % Apply saturations 
            if norm(uDot(2*iii-1:2*iii))>v_max
                udot(2*iii-1:2*iii) = v_max*(uDot(2*iii-1:2*iii))/norm(uDot(2*iii-1:2*iii));
            end
        end
    end
    
    for i=1:N_herders
        if (u(2*i-1)+T*udot(2*i-1))>X_max
            udot(2*i-1) = (X_max-u(2*i-1))/T;
        elseif (u(2*i-1)+T*udot(2*i-1))<X_min
            udot(2*i-1) = (X_min-u(2*i-1))/T;
        end
        if (u(2*i)+T*udot(2*i-1))>Y_max
            udot(2*i) = (Y_max-u(2*i))/T;
        elseif (u(2*i)+T*udot(2*i-1))<Y_min
            udot(2*i) = (Y_min-u(2*i))/T;
        end
    end

    for i=1:M_evaders
        if (x(2*i-1)+T*xdot(2*i-1))>X_max
            xdot(2*i-1) = (X_max-x(2*i-1))/T;
        elseif (x(2*i-1)+T*xdot(2*i-1))<X_min
            xdot(2*i-1) = (X_min-x(2*i-1))/T;
        end
        if (x(2*i)+T*xdot(2*i-1))>Y_max
            xdot(2*i) = (Y_max-x(2*i))/T;
        elseif (x(2*i)+T*xdot(2*i-1))<Y_min
            xdot(2*i) = (Y_min-x(2*i))/T;
        end
    end
    

    %% Send velocities to evaders and herders, apply barrier certs. and map to unicycle dynamics

    % Change the structure of velocities       
    for i=1:M_evaders
        dx(1,i) = xdot(2*i-1);
        dx(2,i) = xdot(2*i);
    end

    for i=1:N_herders
        du(1,i) = udot(2*i-1);
        du(2,i) = udot(2*i);
    end

    XDOT(:,k)= xdot;
    UDOT(:,k)= udot;

    % Fusion them
    dxu = [dx du];

    %Ensure the robots don't collide
    dxu = si_barrier_certificate(dxu, poses(1:2,:));

    % Transform the single-integrator dynamics to unicycle dynamics using a
    % diffeomorphism, which can be found in the utilities
    dxu = si_to_uni_dynamics(dxu, poses);

    % Set velocities of evaders
    r.set_velocities(1:M_evaders+N_herders, dxu);

    %% Apply the step!!
    r.step();
end

disp(end_time/iterations);

r.debug();

% Plot trajectories on the stage, but apart from the Robotarium pannel
figure()

for i=1:N_herders
    plot(H(2*i-1,:),H(2*i,:), '-.b', 'LineWidth', 2)
    hold on
end

for j=1:M_evaders
    plot(PD(2*j-1,:),PD(2*j,:), 'r', 'LineWidth', 2) 
    if j<M_evaders % Inverse evader
        plot(P(2*j-1,:),P(2*j,:), '-.k', 'LineWidth', 2)
        hold on
        plot(x_d(2*j-1),x_d(2*j),'ro', 'MarkerSize',10,'MarkerFaceColor','r')
        hold on
    else % Exponential evader
        plot(P(2*j-1,:),P(2*j,:), '-.g', 'LineWidth', 2)
        hold on
        plot(x_d(2*j-1),x_d(2*j),'ro', 'MarkerSize',10,'MarkerFaceColor','r')
        hold on        
    end
end
title('Arena')
xlabel('x [m]')
ylabel('y [m]')
xlim([X_min, X_max])
ylim([Y_min, Y_max])
set(gca,'FontSize',24)

function f = myfun(theta,a,b,point1,point2)
    f = (a^2-b^2)*cos(theta)*sin(theta) - point1*a*sin(theta) + point2*b*cos(theta);
end

function [A,B] = buildSystem(x,u,theta_e,betaS,tau,sigmaS)    
    % Init matrices with proper dimensions
    A  = zeros(length(x),length(x));
    B  = zeros(length(x),length(u));
    N_herders = floor(length(u)/2);
    
    % Create A(x,u) 
    % For each evader
    for j=1:1
        for i=1:N_herders
            X              = norm(x(2*j-1:2*j)-u(2*i-1:2*i))^3;
            A(2*j-1,2*j-1) = A(2*j-1,2*j-1) + theta_e(j)/X;
            A(2*j,2*j)     = A(2*j,2*j)     + theta_e(j)/X;
        end
    end
    for j=2:2
        var = betaS*theta_e(j);
        for i=1:N_herders
            if norm(x(2*j-1:2*j)-u(2*i-1:2*i))<tau
                var = theta_e(j);
                break;
            end
        end
        for i=1:N_herders
            X              = norm(x(2*j-1:2*j)-u(2*i-1:2*i))^2;
            A(2*j-1,2*j-1) = A(2*j-1,2*j-1) + var*exp(-1/sigmaS^2*X);
            A(2*j,2*j)     = A(2*j,2*j)     + var*exp(-1/sigmaS^2*X);
        end
    end
    
    % Create B(x,u) 
    % For each evader
    for j=1:1
        for i=1:N_herders
            X              = norm(x(2*j-1:2*j)-u(2*i-1:2*i))^3;
            B(2*j-1,2*i-1) = -theta_e(j)/X;
            B(2*j,2*i)     = -theta_e(j)/X;
        end
    end
    for j=2:2
        var = betaS*theta_e(j);
        for i=1:N_herders
            if norm(x(2*j-1:2*j)-u(2*i-1:2*i))<tau
                var = theta_e(j);
                break;
            end
        end
        for i=1:N_herders
            X              = norm(x(2*j-1:2*j)-u(2*i-1:2*i))^2;
            B(2*j-1,2*i-1) = -var*exp(-1/sigmaS^2*X);
            B(2*j,2*i)     = -var*exp(-1/sigmaS^2*X);
        end
    end
end

function f = calculateF(x, u, A, B)  
    f = A*x + B*u;
end

function h = calculateH2(x, xD, f, K, xDdot)
    h= f + K*(x-xD) - xDdot;
end

function td = buildTD(f, fE, K)   
    td = K*(fE-f);
end

function Jx = buildJx2(x,xD,u,theta_e,beta,tau,sigma,K,xDdot) 
    % Preallocate Jacobian
    epsilon = 1e-6;
    Jx      = zeros(length(x),length(x));
    
    % Calculate each column of the Jacobian
    for i=1:length(x)
        v        = zeros(length(x),1);
        v(i)     = epsilon;
        [A,B]    = buildSystem(x+v, u, theta_e, beta, tau, sigma);
        f        = calculateF(x+v, u, A, B);
        h1       = calculateH2(x+v, xD, f, K, xDdot);
        [A,B]    = buildSystem(x-v, u, theta_e, beta, tau, sigma);
        f        = calculateF(x-v, u, A, B);
        h2       = calculateH2(x-v, xD, f, K, xDdot);    
        Jx(:,i)  = (h1 - h2)/epsilon/2;
    end  
end

function Ju = buildJu2(x,xD,u,theta_e,beta,tau,sigma,K,xDdot)
    % Preallocate Jacobian
    epsilon = 1e-6;
    Ju      = zeros(length(x),length(u));
    
    % Calculate each column of the Jacobian
    for i=1:length(u)
        v        = zeros(length(u),1);
        v(i)     = epsilon;
        [A,B]    = buildSystem(x, u+v, theta_e, beta, tau, sigma);
        f        = calculateF(x, u+v, A, B);
        h1       = calculateH2(x, xD, f, K, xDdot);
        [A,B]    = buildSystem(x, u-v, theta_e, beta, tau, sigma);
        f        = calculateF(x, u-v, A, B);
        h2       = calculateH2(x, xD, f, K, xDdot);     
        Ju(:,i)  = (h1 - h2)/epsilon/2;
    end
end

function J = buildJparams(x,u,beta,tau,sigma)   
    % Preallocate Jacobian
    J = zeros(length(x),floor(length(x)/2));
    
    % For each evader
    j=1;
    for i=1:3
        X              = norm(x(2*j-1:2*j)-u(2*j-1:2*j))^3;
        J(2*j-1:2*j,j) = J(2*j-1:2*j,j) + (x(2*j-1:2*j)-u(2*j-1:2*j))/X;                   
    end
    j=2;
    var = beta;
    for i=1:3
        if norm(x(2*j-1:2*j)-u(2*j-1:2*j))<tau
            var = 1;
            break
        end
    end
    for i=1:3
        X               = norm(x(2*j-1:2*j)-u(2*j-1:2*j))^2;
        J(2*j-1:2*j,j)  = J(2*j-1:2*j,j) + var*(x(2*j-1:2*j)-u(2*j-1:2*j))*exp(-1/sigma^2*X);
    end  
end
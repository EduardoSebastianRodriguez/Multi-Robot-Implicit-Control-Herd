% ROBOTARIUM EXPERIMENT
% --------------------------------------------------------------------------------
% 3 herders vs 3 evaders
%
% Evaders behave following: Inverse Model
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

%% Initialize parameters

% Define number of evaders and herders
N_herders = 3;
M_evaders = 3;

% Define desired position for the evaders 
x_d = [0.6 -0.15 -0.2 -0.3 0.2 0.2]';

% Define initial position of the evaders 
x      = [0.7 0.2 -0.4 -0.4 0.0 0.0]';
x_plot = x;
x_prev = x;

% Define initial position of the herders 
u        = [-1.4 0.0 0.0 -0.8 1.2 0.8]';
u_plot   = u;
u_prev   = u;

% Define velocities
xdot   = zeros(2*M_evaders,1);
dposes = zeros(2,M_evaders+N_herders);
u_inici= zeros(2,N_herders);
u_final= zeros(2,N_herders);
dx     = zeros(2,M_evaders);
du     = zeros(2,N_herders);

% Numeric parameters
T        = 0.033; % From Robotarium documentation
end_time = 0.0;
flag     = 0;

% Select the number of iterations for the experiment and stopping threshold
iterations = 2000;
minError   = 0.04;

% Define space of the arena
X_max = 1.6-0.1; % From Robotarium documentation
X_min = -1.6+0.1;% From Robotarium documentation
Y_max = 1.0-0.1; % From Robotarium documentation
Y_min = -1.0+0.1;% From Robotarium documentation

% Define parameters of the evaders' model
theta_r   = 0.02*ones(M_evaders,1);
theta_e   = 0.015*ones(M_evaders,1);
theta_dot = zeros(M_evaders,1);
v_max     = 0.2;

% Define control gains and adaptive gains
K       = 0.25*eye(2*M_evaders);
Kstar   = 50*eye(2*M_evaders);
Kparams = 200*eye(2*M_evaders);

% Define storing variables 
P   = zeros(2*M_evaders,iterations);
PD  = zeros(2*M_evaders,iterations);
H   = zeros(2*N_herders,iterations);
C_r = zeros(M_evaders,iterations);
C_e = zeros(M_evaders,iterations);

%% Init robotarium
initial_conditions = [];
r = Robotarium('NumberOfRobots', M_evaders+N_herders, 'ShowFigure', true, 'InitialConditions', initial_conditions);


%% Go to initial configuration/pose

% Let's retrieve some of the tools we'll need.  We would like a
% single-integrator position controller, a single-integrator barrier
% function, and a mapping from single-integrator to unicycle dynamics
unicycle_barrier_certificate = create_uni_barrier_certificate('SafetyRadius', 0.14,'BarrierGain', 30);
        
%Create controller and some other things
final_goal_points = vertcat(reshape(horzcat(x',u'),[2,N_herders+M_evaders]),[-2.3 0.7 -0.7 0.0 1.6 -2.3]);
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
si_barrier_certificate = create_si_barrier_certificate('SafetyRadius', 0.14,'BarrierGain', 30);
si_to_uni_dynamics = create_si_to_uni_dynamics();
        
%Set desired position for evaders in the robotarium arena
poses_goal        = reshape(x_d,[2,M_evaders]); 

% Iterate for the previously specified number of iterations
for k = 1:iterations
    
    % Record variables
    P(:,k)  = x;
    PD(:,k) = x_d;
    H(:,k)  = u;
    C_r(:,k)= theta_r;
    C_e(:,k)= theta_e;
    
    %% Check if x*
    flag = 0;
    for i=1:M_evaders
        if norm(poses(1:2,i)-poses_goal(:,i))>minError
            flag = 1;
        end
    end
   
    if flag ~= 0 
        
        % Retrieve the most recent poses from the Robotarium.  The time delay is
        % approximately 0.033 seconds
        poses = r.get_poses();

        %% Proposed algorithm

        % Rebuild vector to use the algorithm
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

        % Update the state according to the evaders' model 
        xdot = 0*xdot;
        for j=1:M_evaders
            for i=1:N_herders
                xdot(2*j-1:2*j) = xdot(2*j-1:2*j) + theta_r(j)*( (x(2*j-1:2*j)-u(2*i-1:2*i))/norm(x(2*j-1:2*j)-u(2*i-1:2*i))^3 );
            end
            if norm(xdot(2*j-1:2*j)) > v_max
                xdot(2*j-1:2*j) = v_max*xdot(2*j-1:2*j)/norm(xdot(2*j-1:2*j));
            end
        end

        % Calculate estimated f(x,u) 
        [A,B] = buildSystem(x,u,theta_e);
        f     = calculateF(x,u,A,B);

        % Calculate h(x,u)
        h = calculateH(x,x_d,f,K);

        % Calculate h*
        hD = -Kstar*h;

        % Calculate theta*
        tD = buildTD(xdot,f,Kparams);

        % Calculate Jacobians 
        Jx = buildJx(x,x_d,u,theta_e,K);
        Ju = buildJu(x,x_d,u,theta_e,K);
        Jt = buildJparams(x,u);

        % Calculate udot: udot = pseudoinverse(Ju)*(-Kstar*h - Jx*f)
        % 0.001*I is to ensure invertibility 
        uDot = Ju'/(Ju*Ju'+0.001*eye(2*M_evaders))*(hD - Jx*f); 

        % Calculate params_dot: params_dot = pseudoinverse(J)*(-Kstar*h - Jx*f)
        % 0.001*I is to ensure invertibility 
        tDot = Jt'/(Jt*Jt' + 0.001*eye(2*M_evaders))*(tD - Jx*xdot - Ju*uDot);
        
        % The constant mitigates noises in the adaptation
        theta_e = theta_e + T*tDot*0.005; 

        udot = uDot;
        
        % Apply saturations 
        for i=1:N_herders
            if norm(udot(2*i-1:2*i))>v_max
                udot(2*i-1:2*i) = v_max*(udot(2*i-1:2*i))/norm(udot(2*i-1:2*i));
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

        % Fusion them
        dxu = [dx du];

        %Ensure the robots don't collide
        dxu = si_barrier_certificate(dxu, poses(1:2,:));

        % Transform the single-integrator dynamics to unicycle dynamics using a
        % diffeomorphism, which can be found in the utilities
        dxu = si_to_uni_dynamics(dxu, poses);
        
        % Ensure the robots don't collide
        dxu = unicycle_barrier_certificate(dxu, poses);

        % Set velocities of evaders
        r.set_velocities(1:M_evaders+N_herders, dxu);

        %% Apply the step!!
        r.step();
        
        %% Plot
        if k==1
            for j=1:M_evaders
                plot(x_d(2*j-1),x_d(2*j),'ro', 'MarkerSize',70,'MarkerFaceColor','r')
                hold on
            end
        end
        
    end
end

    
disp(end_time/iterations);

r.debug();

% Plot trajectories on the stage
figure()

for i=1:N_herders
    plot(H(2*i-1,:),H(2*i,:), '-.b', 'LineWidth', 2)
    hold on
end

for j=1:M_evaders
    plot(P(2*j-1,:),P(2*j,:), 'k', 'LineWidth', 2)
    hold on
    plot(x_d(2*j-1),x_d(2*j),'ro', 'MarkerSize',10,'MarkerFaceColor','r')
    hold on
end
title('Arena')
xlabel('x [m]')
ylabel('y [m]')
xlim([X_min, X_max])
ylim([Y_min, Y_max])
set(gca,'FontSize',24)

function [A,B] = buildSystem(x,u,theta_e)    
    % Init matrices with proper dimensions
    A  = zeros(length(x),length(x));
    B  = zeros(length(x),length(u));
    N_herders = floor(length(u)/2);
    M_evaders = floor(length(x)/2);
    
    % Create A(x,u) 
    % For each evader
    for j=1:M_evaders
        for i=1:N_herders
            X              = norm(x(2*j-1:2*j)-u(2*i-1:2*i))^3;
            A(2*j-1,2*j-1) = A(2*j-1,2*j-1) + theta_e(j)/X;
            A(2*j,2*j)     = A(2*j,2*j)     + theta_e(j)/X;
        end
    end
    
    % Create B(x,u) 
    % For each evader
    for j=1:M_evaders
        for i=1:N_herders
            X              = norm(x(2*j-1:2*j)-u(2*i-1:2*i))^3;
            B(2*j-1,2*i-1) = -theta_e(j)/X;
            B(2*j,2*i)     = -theta_e(j)/X;
        end
    end
end

function f = calculateF(x, u, A, B)  
    f = A*x + B*u;
end

function h = calculateH(x, xD, f, K)
    h = f + K*(x-xD);
end

function td = buildTD(f, fE, K)   
    td = K*(fE-f);
end

function Jx = buildJx(x,xD,u,theta_e,K) 
    % Preallocate Jacobian
    epsilon = 1e-6;
    Jx      = zeros(length(x),length(x));
    
    % Calculate each column of the Jacobian
    for i=1:length(x)
        v        = zeros(length(x),1);
        v(i)     = epsilon;
        [A,B]    = buildSystem(x+v, u, theta_e);
        f        = calculateF(x+v, u, A, B);
        h1       = calculateH(x+v, xD, f, K);
        [A,B]    = buildSystem(x-v, u, theta_e);
        f        = calculateF(x-v, u, A, B);
        h2       = calculateH(x-v, xD, f, K);    
        Jx(:,i)  = (h1 - h2)/epsilon/2;
    end  
end

function Ju = buildJu(x,xD,u,theta_e,K)
    % Preallocate Jacobian
    epsilon = 1e-6;
    Ju      = zeros(length(x),length(u));
    
    % Calculate each column of the Jacobian
    for i=1:length(u)
        v        = zeros(length(u),1);
        v(i)     = epsilon;
        [A,B]    = buildSystem(x, u+v, theta_e);
        f        = calculateF(x, u+v, A, B);
        h1       = calculateH(x, xD, f, K);
        [A,B]    = buildSystem(x, u-v, theta_e);
        f        = calculateF(x, u-v, A, B);
        h2       = calculateH(x, xD, f, K);     
        Ju(:,i)  = (h1 - h2)/epsilon/2;
    end
end

function J = buildJparams(x,u)   
    % Preallocate Jacobian
    J = zeros(length(x),floor(length(x)/2));
    N_herders = floor(length(u)/2);
    M_evaders = floor(length(x)/2);
    
    % For each prey
    for j=1:M_evaders
        for i=1:N_herders
            X              = norm(x(2*j-1:2*j)-u(2*j-1:2*j))^3;
            J(2*j-1:2*j,j) = J(2*j-1:2*j,j) + (x(2*j-1:2*j)-u(2*j-1:2*j))/X; 
        end  
    end
end
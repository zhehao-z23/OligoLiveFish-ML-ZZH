% gauss2D is the objecetive function for fitting a tiff image of a gaussian
% beam.  Gauss2D calculates a 2D circular or elliptical gaussian curve and 
% its Jacobian at rectaliner points given by the size of matrix data.

function [F, J] = gauss2D(a,data)

%Inputs:
% a = vector of parameters. The size of a determines the model
%     function returned: size(a)=5 returns circular gaussian with baseline
%     offset, size(a)=6 returns elliptical gaussian with baseline offset,
%     size(a)=7 returns rotated elliptical gaussian. a(1)=offset,
%     a(2)=amplitude, a(3)=x center, a(4)=y center, a(5)= x width (for 
%     circular gaussian x-width = y-width), a(6)= y-width, a(7)= angle theta
%
% data = 2d matrix of image data or 2d matrix which is the same size as
%        image data
%
%Outputs:
%F = 2d matrix of calculated 2D gaussian
%J = jacobian of 2D gaussian
%
% Dependencies: 

% Author: Colin Ingram
% Created: February 2007
% Version: 1.11

% Revisions:
% Version  Date       Author   Description
% 1.10                  BPB:   should allow for non-circular gaussians depending on 
%                              depending on how many input parameters are in a
% 1.11     2009.03.08   CJI:   updated header and comments added warning
%                              for jacobian output 

%% Determine type of 2d gaussian
if ~(size(a,2)==1)
    a = a';  % input was a row vector
   
    if ~(size(a,2)==1) %input is a matrix
        error('vector ''a'' must be a column vector with 5-7 values!');
    end
end

switch size(a,1)
    case 5 %circular gaussian with a baseline offset
        %a(1) is offset
        %a(2) is amplitude
        %a(3) is x center
        %a(4) is y center
        %a(5) is width
        a(6) = a(5);
        a(7) = 0;
    case 6 %elliptical gaussian oriented along x and y
        %a(1) is offset
        %a(2) is amplitude
        %a(3) is x center
        %a(4) is y center
        %a(5) is x width
        %a(6) is y width
        a(7) = 0;
    case 7 %elliptical gaussian turned at angle theta
        %a(1) is offset
        %a(2) is amplitude
        %a(3) is x center
        %a(4) is y center
        %a(5) is x width
        %a(6) is y width   
        %a(7) is angle theta
    otherwise
        error('vector ''a'' must be a column vector with 5-7 values!');
end
        

%% Calculate

sizex = size(data,2);
sizey = size(data,1);
[x,y]= meshgrid(1:sizex,1:sizey);

% 2009.03.08 this seems to do nothing remove?
% theta = a(7);
% sigma_x = a(5);
% sigma_y = a(6);

%% functional form of 2d gaussian from Wikipedia
b = (cos(a(7))^2)/(2*(a(5)^2)) + (sin(a(7))^2)/(2*(a(6)^2));
c = (-(sin(2*a(7)))/(4*(a(5)^2)) + sin(2*a(7))/(4*(a(6)^2))) ;
d = ((sin(a(7))^2)/(2*(a(5)^2)) + (cos(a(7))^2)/(2*(a(6)^2)));
 
fexp = exp( - (b*(x-a(3)).^2 + 2*c*(x-a(3)).*(y-a(4)) + d*(y-a(4)).^2)) ;

F = a(1) + a(2)*fexp;

% 20090105 bpb this calculates the jacobian, doesn't seem to be working
% according to colin
% 2009.03.08 could be fixed just have to calculated out
if nargout > 1
    %     J(1) = 1;
    %     J(2) = fexp;
    %     J(3) = (4*a(2)*(x-a(3))/a(5)^2).*fexp;
    %     J(4) = (4*a(2)*(y-a(4))/a(6)^2).*fexp;
    %     J(5) = (4*a(2)*(x-a(3).^2)/a(5)^3).*fexp;
    J=[];
    warning('Jacobian Output is Broken Please Fix');
end
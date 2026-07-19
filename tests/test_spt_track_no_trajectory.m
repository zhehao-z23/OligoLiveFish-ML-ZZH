function test_spt_track_no_trajectory
% Regression coverage for valid zero-trajectory SPT results.

test_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(test_dir);
addpath(fullfile(repo_root, 'trajectory_extraction', 'pipeline', 'matlab_deps'));

sptpara.mtl = 3;
sptpara.max_disp = 1;
sptpara.trackMem = 0;

% A single detection frame cannot define a temporal trajectory.
[trajlist, traj] = spt_track(sptpara, [10, 10, 1]);
assert(isempty(trajlist));
assert(isempty(traj));

% Detections span consecutive frames, but the candidates cannot link and
% neither one satisfies the minimum retained trajectory length. Legacy code
% reached t_slice(1,3) with an empty trajlist in this case.
poslist = [10, 10, 1; 100, 100, 2];
[trajlist, traj] = spt_track(sptpara, poslist);
assert(isempty(trajlist));
assert(isempty(traj));

fprintf('SPT_EMPTY_TRAJECTORY_REGRESSION_OK\n');
end

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

% Exactly one retained trajectory must be renumbered as ID 1 across every
% row. The legacy single-track boundary used the penultimate row, leaving the
% last row with its old ID and making spt_track iterate over missing IDs.
poslist = [10.0, 10.0, 1; ...
           10.1, 10.1, 2; ...
           10.2, 10.2, 3; ...
           10.3, 10.3, 4];
[trajlist, traj] = spt_track(sptpara, poslist);
assert(~isempty(trajlist));
assert(isequal(unique(trajlist(:,4)), 1));
assert(numel(traj) == 1);
assert(traj(1).length == 3);

fprintf('SPT_EMPTY_TRAJECTORY_REGRESSION_OK\n');
end

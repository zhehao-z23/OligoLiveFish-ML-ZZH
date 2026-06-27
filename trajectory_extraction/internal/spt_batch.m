function spt_batch(tif_path, frame_rate, pixl_um, out_dir)
% spt_batch  Headless single-particle tracking — no GUI required.
%
% frame_rate and pixl_um are passed from run_pipeline.py (read from the
% original full-resolution TIFF, since PIL does not preserve ImageJ
% metadata when saving cropped TIFFs).
%
% out_dir (optional): directory where the .mat file is saved.
%   If omitted, saves next to the input TIFF (original behaviour).
%
% Usage via run_pipeline.py or run_pipeline_v2.py.

fprintf('spt_batch: %s\n', tif_path);
fprintf('  frame_rate = %.4f Hz\n', frame_rate);
fprintf('  pixl       = %.4f um/px\n', pixl_um);

info = imfinfo(tif_path);

% ── Fixed parameters ──────────────────────────────────────────────────────────
sptpara.f_rate            = frame_rate;
sptpara.pixl              = pixl_um;
sptpara.mtl               = 3;
sptpara.dia               = 5;
sptpara.estD              = 0.001;
sptpara.fitmethod         = 0;
sptpara.boxr              = 9;
sptpara.saveFilterImgMode = 1;
sptpara.trackMem          = 3;
sptpara.cell_num          = 1;
sptpara.IntTh             = 0;
sptpara.file              = tif_path;

avgdisp          = sqrt(4 * sptpara.estD / sptpara.f_rate);
sptpara.max_disp = round(3 * avgdisp / sptpara.pixl);

% ── Auto-threshold ────────────────────────────────────────────────────────────
K = 0.5;

first_frame = imread(tif_path, 1, 'Info', info);
filtered    = bpass(single(first_frame), 1, sptpara.dia+2, 0, 'single');
nz          = filtered(filtered > 0);
auto_thresh = mean(nz) + K * std(nz);
sptpara.thresh = auto_thresh;
fprintf('  auto_thresh= %.2f  (mean(nz)+%.1f*std(nz))\n', auto_thresh, K);

fprintf('\n  --- All sptpara ---\n');
fprintf('  f_rate            = %.4f Hz\n',   sptpara.f_rate);
fprintf('  pixl              = %.4f um/px\n', sptpara.pixl);
fprintf('  thresh            = %.4f\n',       sptpara.thresh);
fprintf('  dia               = %d px\n',      sptpara.dia);
fprintf('  boxr              = %d px\n',      sptpara.boxr);
fprintf('  mtl               = %d frames\n',  sptpara.mtl);
fprintf('  estD              = %.4f um^2/s\n',sptpara.estD);
fprintf('  max_disp          = %d px\n',      sptpara.max_disp);
fprintf('  fitmethod         = %d  (0=2DGaussian, 1=centroid)\n', sptpara.fitmethod);
fprintf('  trackMem          = %d frames\n',  sptpara.trackMem);
fprintf('  IntTh             = %d\n',         sptpara.IntTh);
fprintf('  saveFilterImgMode = %d\n',         sptpara.saveFilterImgMode);
fprintf('  cell_num          = %d\n',         sptpara.cell_num);
fprintf('  image size        = %d x %d px\n', info(1).Width, info(1).Height);
fprintf('  -------------------\n\n');

% ── Load image stack ──────────────────────────────────────────────────────────
imcnt = numel(info);
im.rawImg = zeros([info(1).Height, info(1).Width, imcnt], class(first_frame));
im.rawImg(:,:,1) = first_frame;
for j = 2:imcnt
    im.rawImg(:,:,j) = imread(tif_path, j, 'Info', info);
end
im.imageAttr.width  = info(1).Width;
im.imageAttr.height = info(1).Height;
sptpara.size        = [info(1).Width, info(1).Height];

% ── Run SPT pipeline ──────────────────────────────────────────────────────────
[im, poslist] = spt_fndpos(sptpara, im);

if isempty(poslist)
    fprintf('  No particles detected — no trajectories output.\n');
    traj = struct([]);
else
    [~, traj] = spt_track(sptpara, poslist);
    fprintf('  Trajectories found: %d\n', length(traj));
end

% ── Save .mat ────────────────────────────────────────────────────────────────
if (sptpara.saveFilterImgMode == 0) || (sptpara.saveFilterImgMode == 2)
    im = rmfield(im, 'filterImg');
end
im = rmfield(im, 'rawImg');

if nargin < 4 || isempty(out_dir)
    mat_path = [tif_path(1:end-4), '.mat'];
else
    [~, fname, ~] = fileparts(tif_path);
    mat_path = fullfile(out_dir, [fname, '.mat']);
end
save(mat_path, 'im', 'traj', 'sptpara');
fprintf('  Saved: %s\n', mat_path);
end

function spt_batch_anchor_roi(tif_path, roi_mask_path, frame_rate, pixl_um, out_dir, save_filter_images, max_disp_px)
% Run the standard Gaussian SPT detector inside one static irregular ROI.
%
% The ROI gates coarse peak centers through matlab_deps/pkfnd.m. Pixels are
% not zeroed, so bpass and the Gaussian fit still see the original image.

if nargin < 7 || isempty(max_disp_px)
    error(['max_disp_px must be supplied explicitly. Derive it from TIFF ', ...
        'metadata and the declared physical prior before calling this wrapper.']);
end

global SPT_ROI_MASK %#ok<GVMIS>
SPT_ROI_MASK = imread(roi_mask_path) > 0;
cleanup = onCleanup(@() clear_roi_mask());

info = imfinfo(tif_path);
if size(SPT_ROI_MASK, 1) ~= info(1).Height || size(SPT_ROI_MASK, 2) ~= info(1).Width
    error('ROI mask shape does not match input TIFF: mask=%dx%d image=%dx%d', ...
        size(SPT_ROI_MASK, 1), size(SPT_ROI_MASK, 2), ...
        info(1).Height, info(1).Width);
end
if ~any(SPT_ROI_MASK(:))
    error('ROI mask is empty: %s', roi_mask_path);
end

spt_batch(tif_path, frame_rate, pixl_um, out_dir, save_filter_images, max_disp_px);
fprintf('  Static anchor ROI applied: %s\n', roi_mask_path);
clear cleanup
end


function clear_roi_mask()
global SPT_ROI_MASK %#ok<GVMIS>
SPT_ROI_MASK = [];
end

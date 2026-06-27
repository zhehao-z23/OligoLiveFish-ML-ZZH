function varargout = tiffread(filename, varargin)

% tiffread, version 2.8.2 Feb 15, 2011
%
% stack = tiffread;
% stack = tiffread(filename);
% stack = tiffread(filename, indices);
%
% Reads 8,16,32 bits uncompressed grayscale and (some) color tiff files,
% as well as stacks or multiple tiff images, for example those produced
% by metamorph, Zeiss LSM, Andor Solis or NIH-image.
%
% The function can be called with a file name in the current directory,
% or without argument, in which case it pops up a file opening dialog
% to allow for a manual selection of the file.
% If the stacks contains multiples images, reading can be restricted by
% specifying the indices of the desired images (eg. 1:5), or just one index (eg. 2).
%
% The returned value 'stack' is a vector struct containing the images 
% and their meta-data. The length of the vector is the number of images.
% The image pixels values are stored in a field .data, which is a simple
% matrix for gray-scale images, or a cell-array of matrices for color images.
%
% The pixels values are returned in their native (usually integer) format,
% and must be converted to be used in most matlab functions.
%
% Example:
% im = tiffread('spindle.stk');
% imshow( double(im(5).data) );
%
% Only a fraction of the TIFF standard is supported, but you may extend support
% by modifying this file. If you do so, please return your modification to
% F. Nedelec, so that the added functionality can be distributed in the future.
%
% Francois Nedelec, EMBL, Copyright 1999-2010.
% rewriten July 7th, 2004 at Woods Hole during the physiology course.
% last modified January 17, 2011.
% With contributions from:
%   Kendra Burbank for the waitbar
%   Hidenao Iwai for the code to read floating point images,
%   Stephen Lang to be more compliant with PlanarConfiguration
%   Jan-Ulrich Kreft for Zeiss LSM support
%   Elias Beauchanp and David Kolin for additional Metamorph support
%   Jean-Pierre Ghobril for requesting that image indices may be specified
%   Urs Utzinger for the better handling of color images, and LSM meta-data
%   O. Scott Sands for support of GeoTIFF tags
%   Colin Ingram and Benjamin Bratton for adding tags for Andor tiff files
%
% Please, send us feedback/bugs/suggestions to improve this code.
% This software is provided at no cost by a public research institution.
%
% Francois Nedelec
% nedelec (at) embl.de
% Cell Biology and Biophysics, EMBL; Meyerhofstrasse 1; 69117 Heidelberg; Germany
% http://www.embl.org
% http://www.cytosim.org



%Optimization: join adjacent TIF strips: this results in faster reads
consolidateStrips = 1;

%without argument, we ask the user to choose a file:
if nargin < 1
    [filename, pathname] = uigetfile('*.tif;*.stk;*.lsm', 'select image file');
    filename = fullfile(pathname, filename);
end

if (nargin<=1);  indices = 1:10000; end

% parse vargin to support previous versions
switch length(varargin)
    case 0 % nothing to parse
    case 1
        indices = varargin{1};
    case 2
        indices = varargin{1}:varargin{2};
    otherwise
        indices = varargin{1}:varargin{2};
        warning('tiffread:inputArguments',...
            ['tiffread was called with ', num2str(length(varargin)+1),...
            ' input arguments. Recommended number of arguments is 2 or less.']);
end


% not all valid tiff tags have been included, as they are really a lot...
% if needed, tags can easily be added to this code
% See the official list of tags:
% http://partners.adobe.com/asn/developer/pdfs/tn/TIFF6.pdf
%
% the structure IMG is returned to the user, while TIF is not.
% so tags usefull to the user should be stored as fields in IMG, while
% those used only internally can be stored in TIF.
% 
% the structure IMG2 is has additional header information which is added to
% each plane of the image

global TIF;
TIF = [];

%counters for the number of images read and skipped
img_skip  = 0;
img_read  = 1;
hWaitbar  = [];

%% set defaults values :
TIF.SampleFormat     = 1;
TIF.SamplesPerPixel  = 1;
TIF.BOS              = 'ieee-le';          %byte order string

if  isempty(findstr(filename,'.'))
    filename = [filename,'.tif'];
end

TIF.file = fopen(filename,'r','l');
if TIF.file == -1
    stkname = strrep(filename, '.tif', '.stk');
    TIF.file = fopen(stkname,'r','l');
    if TIF.file == -1
        error(['File "',filename,'" not found.']);
    else
        filename = stkname;
    end
end
[s, m] = fileattrib(filename);

% obtain the full file path:
filename = m.Name;

% find the file size in bytes:
% m = dir(filename);
% filesize = m.bytes;


%% read header
% read byte order: II = little endian, MM = big endian
byte_order = fread(TIF.file, 2, '*char');
if ( strcmp(byte_order', 'II') )
    TIF.BOS = 'ieee-le';                                % Intel little-endian format
elseif ( strcmp(byte_order','MM') )
    TIF.BOS = 'ieee-be';
else
    error('This is not a TIFF file (no MM or II).');
end


%% ---- read in a number which identifies file as TIFF format
tiff_id = fread(TIF.file,1,'uint16', TIF.BOS);
if (tiff_id ~= 42)
    error('This is not a TIFF file (missing 42).');
end

%% ---- read the byte offset for the first image file directory (IFD)
TIF.img_pos = fread(TIF.file, 1, 'uint32', TIF.BOS);

while  TIF.img_pos ~= 0 

    clear IMG;
    IMG.filename = filename;
    [pathstr, name, ext] = fileparts(filename);
    IMG.imageName = [name, ext];
    % move in the file to the first IFD
    status = fseek(TIF.file, TIF.img_pos, -1);
    if status == -1
        error('invalid file offset (error on fseek)');
    end

    %disp(strcat('reading img at pos :',num2str(TIF.img_pos)));

    %read in the number of IFD entries
    num_entries = fread(TIF.file,1,'uint16', TIF.BOS);
    %disp(strcat('num_entries =', num2str(num_entries)));

    %read and process each IFD entry
    for i = 1:num_entries

        % save the current position in the file
        file_pos  = ftell(TIF.file);

        % read entry tag
        TIF.entry_tag = fread(TIF.file, 1, 'uint16', TIF.BOS);
        % read entry
        entry = readIFDentry;
        %disp(strcat('reading entry <',num2str(TIF.entry_tag),'>'));

        switch TIF.entry_tag
            case 254
                TIF.NewSubfiletype = entry.val;
            case 256         % image width - number of column
                IMG.width          = entry.val;
            case 257         % image height - number of row
                IMG.height         = entry.val;
                TIF.ImageLength    = entry.val;
            case 258         % BitsPerSample per sample
                TIF.BitsPerSample  = entry.val;
                TIF.BytesPerSample = TIF.BitsPerSample / 8;
                IMG.bits           = TIF.BitsPerSample(1);
                %fprintf('BitsPerSample %i %i %i\n', entry.val);
            case 259         % compression
                if ( entry.val ~= 1 )
                    error(['Compression format ', num2str(entry.val),' not supported.']);
                end
            case 262         % photometric interpretation
                TIF.PhotometricInterpretation = entry.val;
                if ( TIF.PhotometricInterpretation == 3 )
                    warning('tiffread:LookUp', 'Ignoring TIFF look-up table');
                end
            case 269
                IMG.document_name  = entry.val;
            case 270         % comments:
                IMG.info           = entry.val;
            case 271
                IMG.make           = entry.val;
            case 273         % strip offset
                TIF.StripOffsets   = entry.val;
                TIF.StripNumber    = entry.cnt;
                %fprintf('StripNumber = %i, size(StripOffsets) = %i %i\n', TIF.StripNumber, size(TIF.StripOffsets));
            case 274        % orientation, reads it but does not orient the matrix
                switch entry.val
                    case 1
                        TIF.orientationTxt = 'TopLeft';
                        TIF.orientation = 1;
                    case 2
                        TIF.orientationTxt = 'TopRight';
                        TIF.orientation = 2;
                    case 3
                        TIF.orientationTxt = 'BotRight';
                        TIF.orientation = 3;
                    case 4
                        TIF.orientationTxt = 'BotLeft';
                        TIF.orientation = 4;
                    case 5
                        TIF.orientationTxt = 'LeftTop';
                        TIF.orientation = 5;
                    case 6
                        TIF.orientationTxt = 'RightTop';
                        TIF.orientation = 6;
                    case 7
                        TIF.orientationTxt = 'RightBot';
                        TIF.orientation = 7;
                    case 8
                        TIF.orientationTxt = 'LeftBot';
                        TIF.orientation = 8;
                    otherwise
                        TIF.orientationTxt = 'TopLeft';
                        TIF.orientation = 1;
                end
                
            case 277         % sample_per pixel
                TIF.SamplesPerPixel  = entry.val;
                %fprintf('Color image: sample_per_pixel=%i\n',  TIF.SamplesPerPixel);
            case 278         % rows per strip
                TIF.RowsPerStrip   = entry.val;
            case 279         % strip byte counts - number of bytes in each strip after any compressio
                TIF.StripByteCounts= entry.val;
            case 282         % X resolution
                IMG.x_resolution   = entry.val;
            case 283         % Y resolution
                IMG.y_resolution   = entry.val;
            case 284         %planar configuration describe the order of RGB
                TIF.PlanarConfiguration = entry.val;
            case 296         % resolution unit
                IMG.resolution_unit= entry.val;
            case 305         % software
                IMG.software       = entry.val;
            case 306         % datetime
                IMG.datetime       = entry.val;
            case 315
                IMG.artist         = entry.val;
            case 317        %predictor for compression
                if (entry.val ~= 1); error('unsuported predictor value'); end
            case 320         % color map
                IMG.cmap           = entry.val;
                IMG.colors         = entry.cnt/3;
            case 339
                TIF.SampleFormat   = entry.val;
            case 33550       % GeoTIFF ModelPixelScaleTag
                IMG.ModelPixelScaleTag    = entry.val;
            case 33628       %metamorph specific data
                IMG.MM_private1    = entry.val;
            case 33629       %this tag identify the image as a Metamorph stack!
                TIF.MM_stack       = entry.val;
                TIF.MM_stackCnt    = entry.cnt;
            case 33630       %metamorph stack data: wavelength
                TIF.MM_wavelength  = entry.val;
            case 33631       %metamorph stack data: gain/background?
                TIF.MM_private2    = entry.val;
                        
%           % Andor tags
            case 4864       %unknown andor tag
%               IMG2.Tag4864 = entry.val;
                hasAndorHeader = 1;
            case 4865       %unknown andor tag
%                 IMG2.Tag4865        = entry.val;
            case 4866       %unknown andor tag
%                 IMG2.Tag4866        = entry.val;
            case 4867       %unknown andor tag
%                 IMG2.Tag4867        = entry.val;
            case 4868       %unknown andor tag
%                 IMG2.Tag4868        = entry.val;
            case 4869       %temperature in Celsius when stabilized
                if ~(entry.val == -999)
                    IMG2.temperature        = entry.val;
                end
            case 4870       %unknown andor tag
%                 IMG2.Tag4870        = entry.val;
            case 4871       %unknown andor tag
%                 IMG2.Tag4871        = entry.val;
            case 4872       %unknown andor tag
%                 IMG2.Tag4872        = entry.val;
            case 4873       %unknown andor tag
%                 IMG2.Tag4873        = entry.val;
            case 4874       %unknown andor tag
%                 IMG2.Tag4874        = entry.val;
            case 4875       %unknown andor tag
%                 IMG2.Tag4875        = entry.val;
            case 4876       %exposure time in seconds
                IMG2.exposureTime   = entry.val;
            case 4877       %unknown andor tag, always the same as 4878
%                 IMG2.Tag4877        = entry.val;
            case 4878       %Kinetic cycle time
                IMG2.kineticCycleTime       = entry.val;
            case 4879       %number of accumulations
                IMG2.nAccumulations        = entry.val;
            case 4880       %unknown andor tag
%                 IMG2.Tag4880       = entry.val;               
            case 4881       %Acquisition cycle time
                IMG2.acquisitionCycleTime        = entry.val;
            case 4882       %Readout time in seconds, 1/readoutrate
                IMG2.readoutTime        = entry.val;
            case 4883       %unknown andor tag
%                 IMG2.Tag4883        = entry.val;
            case 4884       %unknown andor tag
                if (entry.val == 9)
                    IMG2.isPhotonCounting = 1;
                else
                    IMG2.isPhotonCounting = 0;
                end
            case 4885         %EM DAC level
                IMG2.emDacLevel     = entry.val;                
            case 4886       %unknown andor tag
%                 IMG2.Tag4886        = entry.val;
            case 4887       %unknown andor tag
%                 IMG2.Tag4887        = entry.val;
            case 4888       %unknown andor tag
%                 IMG2.Tag4888       = entry.val;
            case 4889       %unknown andor tag
%                 IMG2.Tag4889        = entry.val;
            case 4890        %number of frames
                IMG2.nFrames       = entry.val;               
            case 4891       %unknown andor tag
%                 IMG2.Tag4891        = entry.val;
            case 4892       %unknown andor tag
%                 IMG2.Tag4892        = entry.val;
            case 4893       %unknown andor tag
%                 IMG2.Tag4893        = entry.val;
            case 4894       %unknown andor tag
%                 IMG2.Tag4894        = entry.val;
            case 4895       %unknown andor tag
%                 IMG2.Tag4895        = entry.val;                
            case 4896         %Horizontally flipped
                IMG2.isFlippedHorizontally        = entry.val;
            case 4897         %Vertically flipped
                IMG2.isFlippedVertically       = entry.val;
            case 4898         %Clockwise rotation
                IMG2.isRotatedClockwise      = entry.val;
            case 4899         %Counterclockwise rotation
                IMG2.isRotatedAnticlockwise       = entry.val;            
            case 4900       %unknown andor tag
%                 IMG2.Tag4900       = entry.val;               
            case 4901       %unknown andor tag
%                 IMG2.Tag4901        = entry.val;
            case 4902       %unknown andor tag
%                 IMG2.Tag4902        = entry.val;
            case 4903       %unknown andor tag
%                 IMG2.Tag4903        = entry.val;
            case 4904       %verticalClockVoltageAmplitude
                IMG2.verticalClockVoltageAmplitude   = entry.val;
            case 4905         %Vertical shift speed in seconds
                IMG2.verticalShiftSpeed       = entry.val;                
            case 4906       %unknown andor tag
%                 IMG2.Tag4906        = entry.val;
            case 4907       %Pre Amp Setting
                IMG2.preAmpSetting        = entry.val;
            case 4908         %Camera Serial Number
                IMG2.serialNumber   = entry.val;
            case 4909       %unknown andor tag
%                 IMG2.Tag4909        = entry.val;
            case 4910       %unknown andor tag
%                 IMG2.Tag4910       = entry.val;               
            case 4911       %Actual camera temperature when not equal to -999
                if ~(entry.val == -999)
                    IMG2.unstabilizedTemperature        = entry.val;
                end
            case 4912         %baseline clamp
                IMG2.isBaselineClamped        = entry.val;
            case 4913         %prescans
                IMG2.nPrescans        = entry.val;
            case 4914         %Model
                IMG2.model          = entry.val;
            case 4915         %chip size, x
                IMG2.chipXSize        = entry.val;                
            case 4916         %chip size, y
                IMG2.chipYSize        = entry.val;
            case 4917       %unknown andor tag
%                 IMG2.Tag4917        = entry.val;
            case 4918       %unknown andor tag
%                 IMG2.Tag4918       = entry.val;
            case 4919       %unknown andor tag
%                 IMG2.Tag4919        = entry.val;  
            case 4920       %unknown andor tag
%                 IMG2.Tag4920       = entry.val;               
            case 4921       %unknown andor tag
%                 IMG2.Tag4921        = entry.val;
            case 4922       %unknown andor tag
%                 IMG2.Tag4922        = entry.val;
            case 4923       %unknown andor tag
%                 IMG2.Tag4923        = entry.val;
            case 4924       %unknown andor tag
%                 IMG2.Tag4924        = entry.val;
            case 4925       %unknown andor tag
%                 IMG2.Tag4925        = entry.val;                
            case 4926       %unknown andor tag
%                 IMG2.Tag4926        = entry.val;
            case 4927       %unknown andor tag
%                 IMG2.Tag4927        = entry.val;
            case 4928       %unknown andor tag
%                 IMG2.Tag4928       = entry.val;
            case 4929       %unknown andor tag
%                 IMG2.Tag4929        = entry.val;                
            case 4930       %unknown andor tag
%                 IMG2.Tag4930       = entry.val;               
            case 4931       %unknown andor tag
%                 IMG2.Tag4931        = entry.val;
            case 4932       %unknown andor tag
%                 IMG2.Tag4932        = entry.val;
            case 4933       %unknown andor tag
%                 IMG2.Tag4933        = entry.val;
            case 4934       %unknown andor tag
%                 IMG2.Tag4934        = entry.val;
            case 4935       %unknown andor tag
%                 IMG2.Tag4935        = entry.val;                
            case 4936       %unknown andor tag
%                 IMG2.Tag4936        = entry.val;
            case 4937       %unknown andor tag
%                 IMG2.Tag4937        = entry.val;
            case 4938       %unknown andor tag
%                 IMG2.Tag4938       = entry.val;
            case 4939       %unknown andor tag
%                 IMG2.Tag4939        = entry.val;         
            case 4940       %unknown andor tag
%                 IMG2.Tag4940       = entry.val;               
            case 4941       %unknown andor tag
%                 IMG2.Tag4941        = entry.val;
            case 4942       %unknown andor tag
%                 IMG2.Tag4942        = entry.val;
            case 4943       %unknown andor tag
%                 IMG2.Tag4943        = entry.val;
            case 4944         %baseline offset
                IMG2.baselineOffset        = entry.val;
            case 4945       %unknown andor tag
%                 IMG2.Tag4945        = entry.val;                
            case 4946       %unknown andor tag
%                 IMG2.Tag4946        = entry.val;
            case 4947       %unknown andor tag
%                 IMG2.Tag4947        = entry.val;
            case 4948       %unknown andor tag
%                 IMG2.Tag4948       = entry.val;
            case 4949       %unknown andor tag, tied to 4935
%                 IMG2.Tag4949        = entry.val;
            case 4950       %unknown andor tag
%                 IMG2.Tag4950        = entry.val;
%           % end of andor tags
            
			case 33922       % GeoTIFF ModelTiePointTag
                IMG.ModelTiePointTag    = entry.val;
            case 34412       % Zeiss LSM data
                LSM_info           = entry.val;
            case 34735       % GeoTIFF GeoKeyDirectory
                IMG.GeoKeyDirTag       = entry.val;
            case 34737       % GeoTIFF GeoASCIIParameters
                IMG.GeoASCII       = entry.val;
            case 42113       % GeoTIFF GDAL_NODATA
                IMG.GDAL_NODATA    = entry.val;
            otherwise
                fprintf( 'Ignored TIFF entry with tag %i (cnt %i)\n', TIF.entry_tag, entry.cnt);
        end
        
        % calculate bounding box  if you've got the stuff
        if isfield(IMG, 'ModelPixelScaleTag') && isfield(IMG, 'ModelTiePointTag') && isfield(IMG, 'height')&& isfield(IMG, 'width'),
            IMG.North=IMG.ModelTiePointTag(5)-IMG.ModelPixelScaleTag(2)*IMG.ModelTiePointTag(2);
            IMG.South=IMG.North-IMG.height*IMG.ModelPixelScaleTag(2);
            IMG.West=IMG.ModelTiePointTag(4)+IMG.ModelPixelScaleTag(1)*IMG.ModelTiePointTag(1);
            IMG.East=IMG.West+IMG.width*IMG.ModelPixelScaleTag(1);
        end

        % move to next IFD entry in the file
        status = fseek(TIF.file, file_pos+12, -1);
        if status == -1
            error('invalid file offset (error on fseek)');
        end
    end

    %Planar configuration is not fully supported
    %Per tiff spec 6.0 PlanarConfiguration irrelevent if SamplesPerPixel==1
    %Contributed by Stephen Lang
    if (TIF.SamplesPerPixel ~= 1) && ( ~isfield(TIF, 'PlanarConfiguration') || TIF.PlanarConfiguration == 1 )
        error('PlanarConfiguration = 1 is not supported');
    end

    %total number of bytes per image:
    PlaneBytesCnt = IMG.width * IMG.height * TIF.BytesPerSample;

    %% try to consolidate the TIFF strips if possible
    
    if consolidateStrips
        %Try to consolidate the strips into a single one to speed-up reading:
        BytesCnt = TIF.StripByteCounts(1);

        if BytesCnt < PlaneBytesCnt

            ConsolidateCnt = 1;
            %Count how many Strip are needed to produce a plane
            while TIF.StripOffsets(1) + BytesCnt == TIF.StripOffsets(ConsolidateCnt+1)
                ConsolidateCnt = ConsolidateCnt + 1;
                BytesCnt = BytesCnt + TIF.StripByteCounts(ConsolidateCnt);
                if ( BytesCnt >= PlaneBytesCnt ); break; end
            end

            %Consolidate the Strips
            if ( BytesCnt <= PlaneBytesCnt(1) ) && ( ConsolidateCnt > 1 )
                %fprintf('Consolidating %i stripes out of %i', ConsolidateCnt, TIF.StripNumber);
                TIF.StripByteCounts = [BytesCnt; TIF.StripByteCounts(ConsolidateCnt+1:TIF.StripNumber ) ];
                TIF.StripOffsets = TIF.StripOffsets( [1 , ConsolidateCnt+1:TIF.StripNumber] );
                TIF.StripNumber  = 1 + TIF.StripNumber - ConsolidateCnt;
            end
        end
    end

    %% read the next IFD address:
    TIF.img_pos = fread(TIF.file, 1, 'uint32', TIF.BOS);
    %if (TIF.img_pos) disp(['next ifd at', num2str(TIF.img_pos)]); end

    if isfield( TIF, 'MM_stack' )

        sel = ( indices <= TIF.MM_stackCnt );
        indices = indices(sel);
        
        if numel(indices) > 1
            hWaitbar = waitbar(0,'Reading images...','Name','TiffRead');
        end

        %this loop reads metamorph stacks:
        for ii = indices

            TIF.StripCnt = 1;
            offset = PlaneBytesCnt * (ii-1);

            %read the image channels
            for c = 1:TIF.SamplesPerPixel
                IMG.data{c} = read_plane(offset, IMG.width, IMG.height, c);
            end

            % print a text timer on the main window, or update the waitbar
            % fprintf('img_read %i img_skip %i\n', img_read, img_skip);
            if ~isempty( hWaitbar )
                waitbar(img_read/numel(indices), hWaitbar);
            end
            
            [ IMG.MM_stack, IMG.MM_wavelength, IMG.MM_private2 ] = splitMetamorph(ii);
            
            stack(img_read) = IMG;
            img_read = img_read + 1;

        end
        break;

    else

        %this part reads a normal TIFF stack:
        
        read_img = any( img_skip+img_read == indices );
        if exist('stack','var')
            if IMG.width ~= stack(1).width || IMG.height ~= stack(1).height
                %setting read_it=0 will skip dissimilar images:
                %comment-out the line below to allow dissimilar stacks
                read_img = 0;
            end
        end
        
        if read_img
            TIF.StripCnt = 1;
            %read the image channels
            for c = 1:TIF.SamplesPerPixel
                IMG.data{c} = read_plane(0, IMG.width, IMG.height, c);
            end

            try
                stack(img_read) = IMG;  % = orderfields(IMG);
                img_read = img_read + 1;
            catch
                fprintf('Tiffread skipped dissimilar image %i\n', img_read+img_skip);
                img_skip = img_skip + 1;
             end
            
            if  all( img_skip+img_read > indices )
                break;
            end

        else
            img_skip = img_skip + 1;
        end

    end
end

%% remove the cell structure if there is always only one channel
flat = 1;
for i = 1:numel(stack)
    if numel(stack(i).data) ~= 1
        flat = 0;
        break;
    end
end

if flat
    for i = 1:numel(stack)
        stack(i).data = stack(i).data{1};
    end
end



%% distribute andor header to all planes.
if not(exist('hasAndorHeader'));
    hasAndorHeader = false;
end

if hasAndorHeader
    IMG2FieldNames = fieldnames(IMG2);
    for i = 1:size(stack,2)
        for j = 1:size(IMG2FieldNames,1)
            stack = setfield(stack, {i}, IMG2FieldNames{j}, ...
                IMG2.( IMG2FieldNames{j} ) );
        end
        stack = setfield(stack, {i}, 'planeNumber', i);
    end
end

% check and see if nFrames is a tag
isNFrames = isfield(stack,'nFrames');

% set nFrames if it doesn't exist
if not(isNFrames);
    nFrames = size(stack,2);
    stack = setfield(stack, {1}, 'nFrames', nFrames);
end

%% distribute the MetaMorph info
if isfield(TIF, 'MM_stack') && isfield(IMG, 'info') && ~isempty(IMG.info)
    MM = parseMetamorphInfo(IMG.info, TIF.MM_stackCnt);
    for i = 1:numel(stack)
        stack(i).MM = MM(i);
    end
end

%% duplicate the LSM info
if exist('LSM_info', 'var')
    for i = 1:numel(stack)
        stack(i).lsm = LSM_info;
    end
end


%% return

if ~ exist('stack', 'var')
    stack = [];
end

%clean-up
fclose(TIF.file);
if ~isempty( hWaitbar )
    delete( hWaitbar );
end

% return values to caller
if nargout>0;
    varargout{1} = stack;
end
if nargout>1;
    varargout{2} = length(stack);
end
if nargout>2;
    warning('tiffread:outputArguments',...
            ['tiffread was called with ', num2str(length(varargin)+1),...
            ' output arguments. Recommended number of arguments is 2 or less.']);
        vargout(3:end) = [];
end
end


%% ===========================================================================

function plane = read_plane(offset, width, height, plane_nb)

global TIF;

%return an empty array if the sample format has zero bits
if ( TIF.BitsPerSample(plane_nb) == 0 )
    plane=[];
    return;
end

%fprintf('reading plane %i size %i %i\n', plane_nb, width, height);

%determine the type needed to store the pixel values:
switch( TIF.SampleFormat )
    case 1
        classname = sprintf('uint%i', TIF.BitsPerSample(plane_nb));
    case 2
        classname = sprintf('int%i', TIF.BitsPerSample(plane_nb));
    case 3
        if ( TIF.BitsPerSample(plane_nb) == 32 )
            classname = 'single';
        else
            classname = 'double';
        end
    otherwise
        error('unsuported TIFF sample format %i', TIF.SampleFormat);
end

% Preallocate a matrix to hold the sample data:
try
    plane = zeros(width, height, classname);
catch
    %compatibility with older matlab versions:
    eval(['plane = ', classname, '(zeros(width, height));']);
end

% Read the strips and concatenate them:
line = 1;
while ( TIF.StripCnt <= TIF.StripNumber )

    strip = read_strip(offset, width, plane_nb, TIF.StripCnt, classname);
    TIF.StripCnt = TIF.StripCnt + 1;

    % copy the strip onto the data
    plane(:, line:(line+size(strip,2)-1)) = strip;

    line = line + size(strip,2);
    if ( line > height )
        break;
    end

end

% Extract valid part of data if needed
if ~all(size(plane) == [width height]),
    plane = plane(1:width, 1:height);
    warning('tiffread:Crop','Cropping data: found more bytes than needed');
end

% transpose the image (otherwise display is rotated in matlab)
plane = plane';

end


%% ================== sub-functions to read a strip ===================

function strip = read_strip(offset, width, plane_nb, stripCnt, classname)

global TIF;

%fprintf('reading strip at position %i\n',TIF.StripOffsets(stripCnt) + offset);
StripLength = TIF.StripByteCounts(stripCnt) ./ TIF.BytesPerSample(plane_nb);

%fprintf( 'reading strip %i\n', stripCnt);
status = fseek(TIF.file, TIF.StripOffsets(stripCnt) + offset, 'bof');
if status == -1
    error('invalid file offset (error on fseek)');
end

bytes = fread( TIF.file, StripLength, classname, TIF.BOS );

if any( length(bytes) ~= StripLength )
    error('End of file reached unexpectedly.');
end

strip = reshape(bytes, width, StripLength / width);

end


%% ==================sub-functions that reads an IFD entry:===================


function [nbBytes, matlabType] = convertType(tiffType)
switch (tiffType)
    case 1
        nbBytes=1;
        matlabType='uint8';
    case 2
        nbBytes=1;
        matlabType='uchar';
    case 3
        nbBytes=2;
        matlabType='uint16';
    case 4
        nbBytes=4;
        matlabType='uint32';
    case 5
        nbBytes=8;
        matlabType='uint32';
    case 7
        nbBytes=1;
        matlabType='uchar';
    case 11
        nbBytes=4;
        matlabType='float32';
    case 12
        nbBytes=8;
        matlabType='float64';
    otherwise
        error('tiff type %i not supported', tiffType)
end
end

%% ==================sub-functions that reads an IFD entry:===================

function  entry = readIFDentry()

global TIF;
entry.tiffType = fread(TIF.file, 1, 'uint16', TIF.BOS);
entry.cnt      = fread(TIF.file, 1, 'uint32', TIF.BOS);
%disp(['tiffType =', num2str(entry.tiffType),', cnt = ',num2str(entry.cnt)]);

[ entry.nbBytes, entry.matlabType ] = convertType(entry.tiffType);

if entry.nbBytes * entry.cnt > 4
    %next field contains an offset:
    offset = fread(TIF.file, 1, 'uint32', TIF.BOS);
    %disp(strcat('offset = ', num2str(offset)));
    status = fseek(TIF.file, offset, -1);
    if status == -1
        error('invalid file offset (error on fseek)');
    end

end


if TIF.entry_tag == 33629   % metamorph 'rationals'
    entry.val = fread(TIF.file, 6*entry.cnt, entry.matlabType, TIF.BOS);
elseif TIF.entry_tag == 34412  %TIF_CZ_LSMINFO
    entry.val = readLSMinfo;
else
    if entry.tiffType == 5
        entry.val = fread(TIF.file, 2*entry.cnt, entry.matlabType, TIF.BOS);
    else
        entry.val = fread(TIF.file, entry.cnt, entry.matlabType, TIF.BOS);
    end
end

if ( entry.tiffType == 2 );
    entry.val = char(entry.val');
end

end


%% =============distribute the metamorph infos to each frame:
function [MMstack, MMwavelength, MMprivate2] = splitMetamorph(imgCnt)

global TIF;

MMstack = [];
MMwavelength = [];
MMprivate2 = [];

if TIF.MM_stackCnt == 1
    return;
end

left  = imgCnt - 1;

if isfield( TIF, 'MM_stack' )
    S = length(TIF.MM_stack) / TIF.MM_stackCnt;
    MMstack = TIF.MM_stack(S*left+1:S*left+S);
end

if isfield( TIF, 'MM_wavelength' )
    S = length(TIF.MM_wavelength) / TIF.MM_stackCnt;
    MMwavelength = TIF.MM_wavelength(S*left+1:S*left+S);
end

if isfield( TIF, 'MM_private2' )
    S = length(TIF.MM_private2) / TIF.MM_stackCnt;
    MMprivate2 = TIF.MM_private2(S*left+1:S*left+S);
end

end


%% %%  Parse the Metamorph camera info tag into respective fields
% EVBR 2/7/2005, FJN Dec. 2007
function mm = parseMetamorphInfo(info, cnt)

info   = regexprep(info, '\r\n|\o0', '\n');
parse  = textscan(info, '%s %s', 'Delimiter', ':');
tokens = parse{1};
values = parse{2};

first = char(tokens(1,1));

k = 0;
mm = struct('Exposure', zeros(cnt,1));
for i=1:size(tokens,1)
    tok = char(tokens(i,1));
    val = char(values(i,1));
    %fprintf( '"%s" : "%s"\n', tok, val);
    if strcmp(tok, first)
        k = k + 1;
    end
    if strcmp(tok, 'Exposure')
        [v, c, e, pos] = sscanf(val, '%i');
        unit = val(pos:length(val));
        %return the exposure in milli-seconds
        switch( unit )
            case 'ms'
                mm(k).Exposure = v;
            case 's'
                mm(k).Exposure = v * 1000;
            otherwise
                warning('tiffread2:Unit', ['Exposure unit "',unit,'" not recognized']);
                mm(k).Exposure = v;
        end
    else
        switch tok
            case 'Binning'
                % Binning: 1 x 1 -> [1 1]
                mm(k).Binning = sscanf(val, '%d x %d')';
            case 'Region'
                mm(k).Region = sscanf(val, '%d x %d, offset at (%d, %d)')';
            otherwise
                field  = regexprep(tok, ' ', '');
                if strcmp(val, 'Off')
                    eval(['mm(k).',field,'=0;']);
                elseif strcmp(val, 'On')
                    eval(['mm(k).',field,'=1;']);
                elseif isstrprop(val,'digit')
                    eval(['mm(k).',field,'=str2num(val)'';']);
                else
                    eval(['mm(k).',field,'=val;']);
                end
        end
    end
end

end

%% ==============partial-parse of LSM info:

function R = readLSMinfo()

% Read part of the LSM info table version 2
% this provides only very partial information, since the offset indicate that
% additional data is stored in the file
global TIF;

R.MagicNumber            = sprintf('0x%09X',fread(TIF.file, 1, 'uint32', TIF.BOS));
StructureSize          = fread(TIF.file, 1, 'uint32', TIF.BOS);
R.DimensionX             = fread(TIF.file, 1, 'uint32', TIF.BOS);
R.DimensionY             = fread(TIF.file, 1, 'uint32', TIF.BOS);
R.DimensionZ             = fread(TIF.file, 1, 'uint32', TIF.BOS);
R.DimensionChannels      = fread(TIF.file, 1, 'uint32', TIF.BOS);
R.DimensionTime          = fread(TIF.file, 1, 'uint32', TIF.BOS);
R.IntensityDataType      = fread(TIF.file, 1, 'uint32', TIF.BOS);
R.ThumbnailX             = fread(TIF.file, 1, 'uint32', TIF.BOS);
R.ThumbnailY             = fread(TIF.file, 1, 'uint32', TIF.BOS);
R.VoxelSizeX             = fread(TIF.file, 1, 'float64', TIF.BOS);
R.VoxelSizeY             = fread(TIF.file, 1, 'float64', TIF.BOS);
R.VoxelSizeZ             = fread(TIF.file, 1, 'float64', TIF.BOS);
R.OriginX                = fread(TIF.file, 1, 'float64', TIF.BOS);
R.OriginY                = fread(TIF.file, 1, 'float64', TIF.BOS);
R.OriginZ                = fread(TIF.file, 1, 'float64', TIF.BOS);
R.ScanType               = fread(TIF.file, 1, 'uint16', TIF.BOS);
R.SpectralScan           = fread(TIF.file, 1, 'uint16', TIF.BOS);
R.DataType               = fread(TIF.file, 1, 'uint32', TIF.BOS);
OffsetVectorOverlay    = fread(TIF.file, 1, 'uint32', TIF.BOS);
OffsetInputLut         = fread(TIF.file, 1, 'uint32', TIF.BOS);
OffsetOutputLut        = fread(TIF.file, 1, 'uint32', TIF.BOS);
OffsetChannelColors    = fread(TIF.file, 1, 'uint32', TIF.BOS);
R.TimeInterval           = fread(TIF.file, 1, 'float64', TIF.BOS);
OffsetChannelDataTypes = fread(TIF.file, 1, 'uint32', TIF.BOS);
OffsetScanInformation  = fread(TIF.file, 1, 'uint32', TIF.BOS);
OffsetKsData           = fread(TIF.file, 1, 'uint32', TIF.BOS);
OffsetTimeStamps       = fread(TIF.file, 1, 'uint32', TIF.BOS);
OffsetEventList        = fread(TIF.file, 1, 'uint32', TIF.BOS);
OffsetRoi              = fread(TIF.file, 1, 'uint32', TIF.BOS);
OffsetBleachRoi        = fread(TIF.file, 1, 'uint32', TIF.BOS);
OffsetNextRecording    = fread(TIF.file, 1, 'uint32', TIF.BOS);

% There are more information stored in this table, which is not read here


%read real acquisition times:
if ( OffsetTimeStamps > 0 )
    
    status = fseek(TIF.file, OffsetTimeStamps, -1);
    if status == -1
        error('error on fseek');
    end
    
    StructureSize          = fread(TIF.file, 1, 'int32', TIF.BOS);
    NumberTimeStamps       = fread(TIF.file, 1, 'int32', TIF.BOS);
    for i=1:NumberTimeStamps
        R.TimeStamp(i)       = fread(TIF.file, 1, 'float64', TIF.BOS);
    end
    
    %calculate elapsed time from first acquisition:
    R.TimeOffset = R.TimeStamp - R.TimeStamp(1);
    
end


end



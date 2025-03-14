function C = read_arma_binary(filename)
    fid = fopen(filename, 'rb');
    if fid == -1
        error('Can not open file %s', filename);
    end

    % 1️. read ASCII header
    header_type = fgetl(fid);
    [dataType, isComplex] = get_data_type_from_header(header_type);
    if dataType=="Unsupported header"
        fclose(fid);
        error('%s', header_type);
    end

    % 2. Parse size information (separated by space)
    header_shape = fgetl(fid);
    dims = sscanf(header_shape, '%d %d %d');
    if numel(dims) ~= 3
        fclose(fid);
        error('can not parse dimension information: %s', header_shape);
    end
    sizeX = dims(1);
    sizeY = dims(2);
    sizeZ = dims(3);

    % 3. Read data part (encoded by dataType)
    if isComplex % for complex numbers
        rawData = fread(fid, 2*sizeX * sizeY * sizeZ, dataType);
        rawData = rawData(1:2:end) + 1i*rawData(2:2:end);
        dataType = dataType + " complex"; % complex number types
    else
        rawData = fread(fid, sizeX * sizeY * sizeZ, dataType);
    end
    fclose(fid);

    % 4️. Reshape to 3D matrix
    C = reshape(rawData, [sizeX, sizeY, sizeZ]);

    % 5️. Display results
    fprintf('Successfully read %dx%dx%d cube (%s)\n', sizeX, sizeY, sizeZ, dataType);
end

function [dataType, isComplex] = get_data_type_from_header(header)
    isComplex = false;    % Indicates that the data is complex
    switch header
        case 'ARMA_CUB_BIN_IU001'
            dataType = 'uint8';  % Unsigned 8-bit integer
        case 'ARMA_CUB_BIN_IS001'
            dataType = 'int8';   % Signed 8-bit integer
        case 'ARMA_CUB_BIN_IU002'
            dataType = 'uint16'; % Unsigned 16-bit integer
        case 'ARMA_CUB_BIN_IS002'
            dataType = 'int16';  % Signed 16-bit integer
        case 'ARMA_CUB_BIN_IU004'
            dataType = 'uint32'; % Unsigned 32-bit integer
        case 'ARMA_CUB_BIN_IS004'
            dataType = 'int32';  % Signed 32-bit integer
        case 'ARMA_CUB_BIN_IU008'
            dataType = 'uint64'; % Unsigned 64-bit integer
        case 'ARMA_CUB_BIN_IS008'
            dataType = 'int64';  % Signed 64-bit integer
        case 'ARMA_CUB_BIN_FN004'
            dataType = 'single'; % Single precision float
        case 'ARMA_CUB_BIN_FN008'
            dataType = 'double'; % Double precision float
        case 'ARMA_CUB_BIN_FC008'
            dataType = 'single'; % Single precision floating-point
            isComplex = true;    % Indicates that the data is complex
        case 'ARMA_CUB_BIN_FC016'
            dataType = 'double'; % Double precision floating-point
            isComplex = true;    % Indicates that the data is complex
        otherwise
            dataType = 'Unsupported header';
    end
end

import cv2
import numpy as np
from numpy.polynomial import Polynomial
from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.signal import argrelextrema

def segment_rpe_layer(bscan, params, medline):
    """
    Segments the RPE - retinal pigment epithelium layer from a B-scan image.

    Args:
        bscan (numpy.ndarray): Unnormalized B-scan image.
        params (dict): Parameter dictionary for segmentation.
        medline (numpy.ndarray): Reference line for normalization.

    Returns:
        numpy.ndarray: Automated segmentation of the RPE.
    """
    # 1) Normalize the intensity values
    # Save the bscan image before normalization
    cv2.imwrite(f"./tests/bscan_before_normalization.png", bscan)

    # Inspect bscan data
    print('bscan min:', np.min(bscan), 'max:', np.max(bscan))
    print('bscan mean:', np.mean(bscan), 'std:', np.std(bscan))

    # Check if normalization is possible
    if np.max(bscan) == np.min(bscan):
        print("bscan has constant value. Cannot normalize.")
    else:
        # Apply logarithmic scaling if necessary
        if np.max(bscan) - np.min(bscan) > 100:
            bscan = np.log1p(bscan)
            print('After log1p transformation:')
            print('bscan min:', np.min(bscan), 'max:', np.max(bscan))

        # Normalize to range [0, 255]
        # bscan_normalized = 255 * (bscan - np.min(bscan)) / (np.max(bscan) - np.min(bscan))
        # bscan_normalized = bscan_normalized.astype(np.uint8)

        # Normalize the image to [0, 1]
        bscan_min = np.min(bscan)
        bscan_max = np.max(bscan)
        bscan_normalized = (bscan - bscan_min) / (bscan_max - bscan_min)

        # # Display the image
        # cv2.imshow('Normalized Image', bscan_normalized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # Scale to [0, 255] and convert to uint8
    bscan_to_save = (bscan_normalized * 255).astype(np.uint8)
    cv2.imwrite(f"./tests/bscan_after_normalization.png", bscan_to_save)
    sn_bscan = split_normalize(bscan_normalized, params, medline)
    cv2.imwrite(f"./tests/bscan_after_split_normalize.png", sn_bscan[0])
    #print(sn_bscan)
    # print(type(sn_bscan))
    # # Ensure sn_bscan is a NumPy array
    # sn_bscan = np.array(sn_bscan)

    # # Normalize and convert to uint8 if it's not already
    # if sn_bscan.dtype != np.uint8:
    #     sn_bscan = (sn_bscan * 255).astype(np.uint8)
    # Extract the secondary array from the tuple
    #sn_bscan_array = sn_bscan[0]
    #cv2.imwrite(f"./bscan_after_split_normalize.png", sn_bscan_array)
    sn_bscan = remove_bias(sn_bscan, params)
    cv2.imwrite(f"./bscan_after_remove_bias.png", sn_bscan)

    # 2) Noise removal using median filtering
    sn_bscan = medfilt2d(sn_bscan, params['RPECIRC_SEGMENT_MEDFILT1'])
    sn_bscan = medfilt2d(sn_bscan, params['RPECIRC_SEGMENT_MEDFILT2'])

    # 3) Edge detection along A-scans
    rpe = extrema_finder(image=sn_bscan, num_extrema=params,  mode='abs')
    # print(f"{type(rpe)} {params['RPECIRC_SEGMENT_LINESWEETER1']}")
    # # Validate that rpe['min'] is a 1D array
    # #rpe_min = np.asarray(rpe['min'])
    # rpe_min = np.hstack(rpe['min'])
    # #rpe_max = np.asarray(rpe['max'])
    # rpe_max = np.hstack(rpe['max'])
    # if rpe_min.ndim != 1 or rpe_max.ndim != 1:
    #     raise ValueError(f"Expected rpe['min'] and rpe['max'] to be a 1D array, but got an arrays with shape {rpe_min.shape} {rpe_max.shape}")

    #rpe = line_sweeter(rpe['min'], params['RPECIRC_SEGMENT_LINESWEETER1'])
    #rpe = line_sweeter(rpe['max'], params['RPECIRC_SEGMENT_LINESWEETER1'])

    # 4) Polynomial fitting
    # x = np.arange(bscan.shape[1])
    # p = np.polyfit(x, rpe, params['RPECIRC_SEGMENT_POLYNUMBER'])
    # rpe_poly = np.polyval(p, x)
    # rpe_poly = np.round(rpe_poly)
    # dist = np.abs(rpe_poly - rpe)
    # rpe[dist > params['RPECIRC_SEGMENT_POLYDIST']] = 0

    # 5) Remove blood vessel regions and final smoothing
    # idx_bv = find_blood_vessels(sn_bscan, params, rpe)
    # rpe[idx_bv] = 0
    # rpe_auto = line_sweeter(rpe, params['RPECIRC_SEGMENT_LINESWEETER2'])

    return rpe

def line_sweeter(line, window_size):
    """
    Applies median filtering to smooth the input line.

    Args:
        line (numpy.ndarray): 1D array representing the line to be smoothed.
        window_size (int): Size of the median filter window. Must be an odd integer.

    Returns:
        numpy.ndarray: Smoothed line.
    """

    # Check if line is a list of lists or contains elements of different lengths or types
    if isinstance(line, list) and any(isinstance(i, list) for i in line):
        raise ValueError(f"Expected a 1D array, but got a list of lists or elements of different lengths/types {type(line)}")

    # Validate that line is a 1D array
    line = np.asarray(line)
    if line.ndim != 1:
        raise ValueError(f"Expected a 1D array, but got an array with shape {line.shape}")

    # Ensure window_size is odd
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    if window_size % 2 == 0:
        window_size += 1  # Ensure window_size is odd
    
    if type(line) == dict:
        print(line.keys())

    return medfilt(line, kernel_size=window_size)

def extrema_finderNEW(data, num_extrema=2, mode='max'):
    extrema = {'max': [[] for _ in range(data.shape[1])],
              'min': [[] for _ in range(data.shape[1])]}
    
    for j in range(data.shape[1]):
        column = data[:, j]
        if mode == 'max':
            indices = argrelextrema(column, np.greater)[0]
            values = column[indices]
            sorted_indices = indices[np.argsort(values)[::-1]]
            extrema['max'][j] = values[np.argsort(values)[::-1]].tolist()[:num_extrema]
        elif mode == 'min':
            indices = argrelextrema(column, np.less)[0]
            values = column[indices]
            sorted_indices = indices[np.argsort(values)]
            extrema['min'][j] = values[np.argsort(values)].tolist()[:num_extrema]
        else:
            raise ValueError("Invalid mode. Use 'max' or 'min'.")
    
    return extrema

def extrema_finder(image: np.ndarray, num_extrema: int, mode: str = 'abs', lineregion: tuple = None) -> dict:
    """
    Finds the top `num_extrema` maximum and minimum values in each row of the image.

    Parameters:
        image (np.ndarray): 2D array where each row represents a different set of data.
        num_extrema (int): Number of top extrema to find in each row.
        mode (str): Determines the mode for finding extrema. Options include:
            - 'abs': All extrema are detected based on absolute values (default).
            - 'min': Only minima extrema are detected.
            - 'max': Only maxima extrema are detected.
            - 'pos': Extrema are sorted by their position in the A-Scan.
        lineregion (tuple): Defines the region within each row to search for extrema.
                            It is a tuple (start, end) specifying the line region.

    Returns:
        dict: Dictionary containing lists of maximum and minimum extrema values for each row.
              Format:
              {
                  'max': [[max1_row1, max2_row1, ...], [max1_row2, ...], ...],
                  'min': [[min1_row1, min2_row1, ...], [min1_row2, ...], ...]
              }
    """
    # Set default lineregion if not provided
    if lineregion is None:
        lineregion = (0, image.shape[1])

    max_extrema = []
    min_extrema = []

    for row in image:    
        if len(lineregion) != 2:
            raise ValueError(f"lineregion must contain exactly 2 elements, but got {len(lineregion)} elements.")

        start, end = lineregion
        segment = row[start:end]

        # Find maxima and minima indices using find_peaks
        peak_indices, _ = find_peaks(segment)
        trough_indices, _ = find_peaks(-segment)

        # Manually check the first and last elements for maxima
        if len(segment) >= 2:
            if segment[0] > segment[1]:
                peak_indices = np.insert(peak_indices, 0, 0)
            if segment[-1] > segment[-2]:
                peak_indices = np.append(peak_indices, len(segment) - 1)

        # Manually check the first and last elements for minima
        if len(segment) >= 2:
            if segment[0] < segment[1]:
                trough_indices = np.insert(trough_indices, 0, 0)
            if segment[-1] < segment[-2]:
                trough_indices = np.append(trough_indices, len(segment) - 1)

        # Get maxima and minima values
        max_values = segment[peak_indices]
        min_values = segment[trough_indices]

        # Sort and select top extrema
        selected_max = []
        selected_min = []

        #print(type(mode))

        if isinstance(mode, (list, np.ndarray)):
             print(mode)

        if mode in ['abs', 'max', 'min']:
            # Sort maxima in descending order and select top unique
            sorted_max_indices = np.argsort(-max_values)
            for idx in sorted_max_indices:
                val = max_values[idx]
                if val not in selected_max:
                    selected_max.append(int(val))
                if len(selected_max) == num_extrema:
                    break

        if mode in ['abs', 'max', 'min']:
            # Sort minima in ascending order and select top unique
            sorted_min_indices = np.argsort(min_values)
            for idx in sorted_min_indices:
                val = min_values[idx]
                if val not in selected_min:
                    selected_min.append(int(val))
                if len(selected_min) == num_extrema:
                    break

        max_extrema.append(selected_max)
        min_extrema.append(selected_min)

    return {
        'max': max_extrema,
        'min': min_extrema
    }

def find_low_pos_near_lmax(octimg, g, lmax, sigma2):
    """
    Finds the minima in the smoothed image near the detected maxima.

    Args:
        octimg (numpy.ndarray): Original OCT B-scan image.
        g (numpy.ndarray): Smoothed and normalized image.
        lmax (numpy.ndarray): Array of shape (num_extrema, num_cols) with maxima positions.
        sigma2 (float): Sigma value for the second Gaussian smoothing.

    Returns:
        numpy.ndarray: 1D array containing y-indices of the minima per column.
    """
    num_cols = g.shape[1]
    lmin = np.zeros(num_cols, dtype=int)

    for j in range(num_cols):
        # Extract maxima positions for the current column
        maxima = lmax[:, j]
        maxima = maxima[maxima > 0]  # Exclude zero entries

        if len(maxima) == 0:
            # No maxima found; find the global minimum
            lmin[j] = np.argmin(g[:, j])
        else:
            min_max = maxima.min()
            max_max = maxima.max()

            if min_max < max_max:
                search_start = min_max
                search_end = max_max
            else:
                search_start = max_max
                search_end = min_max

            # Handle edge cases where search window might be invalid
            if search_start == search_end:
                search_start = max(0, search_start - 1)
                search_end = min(g.shape[0], search_end + 1)

            region = g[search_start:search_end, j]

            if len(region) == 0:
                lmin[j] = search_start  # Assign the start position if region is empty
            else:
                local_min = np.argmin(region)
                lmin[j] = search_start + local_min

    return lmin

def find_medline(octimg, params):
    # Convert to float32 if necessary
    if octimg.dtype == np.float16:
        octimg = octimg.astype(np.float32)
        print("Converted octimg from float16 to float32 for gaussian_filter.")
    elif not np.issubdtype(octimg.dtype, np.floating):
        octimg = octimg.astype(np.float32)
        print("Converted octimg to float32 for gaussian_filter.")

    mindist = params['MEDLINE_MINDIST']
    sigma1 = params['MEDLINE_SIGMA1']
    sigma2 = params['MEDLINE_SIGMA2']
    linesweeter_window = params['MEDLINE_LINESWEETER']
    
    # Step 1: First Gaussian smoothing
    gsize1 = int(np.floor(sigma1 * 1.5))
    if gsize1 % 2 == 0:
        gsize1 += 1  # Ensure gsize1 is odd
    truncate1 = (gsize1 - 1) / (2.0 * sigma1)
    g = gaussian_filter(octimg, sigma=sigma1, mode='reflect', truncate=truncate1)
    g = g / np.max(g)  # Normalize the smoothed image
    
    # Step 2: Find up to 2 maxima per column
    lmax_dict = extrema_finder(image=g, num_extrema=2, mode='max')
    
    # Convert lmax to a NumPy array with shape (2, number_of_columns)
    num_columns = g.shape[1]
    lmax = np.zeros((2, num_columns), dtype=int)
    
    for j in range(num_columns):
        maxima = lmax_dict.get(j, [])
        for i in range(min(len(maxima), 2)):
            lmax[i, j] = maxima[i]
    
    # Debug: Print the shape and contents of lmax
    print(f"lmax shape: {lmax.shape}")
    print(f"lmax contents: {lmax}")
    
    # Step 3: Remove maxima that are too close to each other
    for j in range(num_columns):
        if lmax.shape[0] > 1 and abs(lmax[1, j] - lmax[0, j]) < mindist:
            lmax[:, j] = 0  # Reset to zero if too close
    
    # Step 4: Apply linesweeter smoothing to maxima
    for i in range(lmax.shape[0]):
        lmax[i, :] = line_sweeter(lmax[i, :], linesweeter_window)
    lmax = np.round(lmax).astype(int)
    
    # Step 5: Second Gaussian smoothing
    gsize2 = int(np.floor(sigma2 * 1.5))
    if gsize2 % 2 == 0:
        gsize2 += 1  # Ensure gsize2 is odd
    truncate2 = (gsize2 - 1) / (2.0 * sigma2)
    g = gaussian_filter(octimg, sigma=sigma2, mode='reflect', truncate=truncate2)
    
    # Step 6: Find minima near the detected maxima
    lmin = extrema_finder(image=g, num_extrema=1, mode='min')
    lmin = find_low_pos_near_lmax(octimg, g, lmax, sigma2)
    
    # Step 7: Apply linesweeter smoothing to minima to obtain medline
    medline = line_sweeter(lmin, linesweeter_window)
    return medline

def split_normalizeNew(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        return image
    normalized = (image - min_val) / (max_val - min_val)
    return normalized

def split_normalize(octimg, params, mode='ipsimple opsimple soft', medline=None):
    """
    Normalize an OCT B-scan differently in the inner and outer parts.

    Args:
        octimg (np.ndarray): Input B-scan image.
        params (dict): Parameter dictionary.
        mode (str): Normalization modes.
        medline (np.ndarray): Medline indices.

    Returns:
        noctimg (np.ndarray): Normalized OCT image.
        medline (np.ndarray): Medline indices.
    """
    cutoff = params.get('SPLITNORMALIZE_CUTOFF', 2.0)
    cutoff = params.get('SPLITNORMALIZE_CUTOFF', 2.0)
    print(f"Normalization mode: {mode}")
    print(f"Cutoff value: {cutoff}")

    if medline is None:
        medline = find_medline(octimg, params)
        medline = medfilt2d(medline, kernel_size=5)
        medline = np.floor(medline).astype(int)
        medline[medline < 1] = 1
        print("Processed medline:", medline)

    noctimg = octimg.copy()
    assert np.all(medline < noctimg.shape[0]), "medline contains out-of-bounds indices."
    num_cols = octimg.shape[1]
    maxIP = np.zeros(num_cols, dtype=np.float64)
    minIP = np.zeros(num_cols, dtype=np.float64)
    maxOP = np.zeros(num_cols, dtype=np.float64)
    minOP = np.zeros(num_cols, dtype=np.float64)

    if 'ipnearmax' not in mode:
        meanVal = np.zeros(num_cols)
    else:
        meanVal = np.zeros(num_cols)
        for i in range(num_cols):
            sorter = np.sort(octimg[:medline[i], i])
            meanVal[i] = np.mean(sorter)
        meanVal = meanVal * cutoff - np.mean(meanVal)
        meanVal[meanVal < 0] = 0
        for i in range(num_cols):
            # Ensure medline[i] does not exceed the maximum index
            start_index = min(medline[i], octimg.shape[0] - 1)
            end_index = min(end_index, octimg.shape[0] - 1)
            
            # Slices for minIP and maxIP
            slice_maxIP = octimg[:end_index, i]
            if slice_maxIP.size > 0:
                minIP[i] = np.min(slice_maxIP)
            else:
                minIP[i] = np.nan  # or another appropriate default value
            
            # Slices for minOP and maxOP
            slice_minOP = octimg[start_index:end_index, i]
            if slice_minOP.size > 0:
                minOP[i] = np.min(slice_minOP)
                maxOP[i] = np.max(slice_minOP)
            else:
                minOP[i] = np.nan  # or another appropriate default value
                maxOP[i] = np.nan  # or another appropriate default value
            
            # Calculate the index of the minimum value in slice_maxIP
            if slice_maxIP.size > 0:
                current_min = np.argmin(slice_maxIP)
            else:
                current_min = -1  # Indicates no valid index
            
            if current_min != -1 and current_min < octimg.shape[0]:
                min_value = octimg[current_min, i]
            else:
                min_value = np.nan  # or handle appropriately
                print(f"Warning: min index {current_min} out of bounds for axis 0 with size {octimg.shape[0]}")

    if 'ipnonlin' in mode:
        maxIP = maxIP - meanVal + minIP

    if 'soft' in mode:
        maxIP = medfilt(maxIP, kernel_size=5)
        minIP = medfilt(minIP, kernel_size=5)
        maxOP = medfilt(maxOP, kernel_size=5)
        minOP = medfilt(minOP, kernel_size=5)
        print("maxIP after 'soft':", maxIP)
        print("minIP after 'soft':", minIP)
        print("maxOP after 'soft':", maxOP)
        print("minOP after 'soft':", minOP)

    ipDiff = maxIP - minIP
    ipDiff[ipDiff == 0] = 1  # Prevent division by zero
    print("ipDiff min:", ipDiff.min(), "max:", ipDiff.max())
    print("minIP min:", minIP.min(), "max:", minIP.max())
    if 'ipbscan' in mode:
        minIP[:] = np.min(minIP)
        maxIP[:] = np.max(minIP)
        ipDiff = maxIP - minIP
        print("minIP after 'ipbscan':", minIP)
        print("maxIP after 'ipbscan':", maxIP)
        print("ipDiff after 'ipbscan':", ipDiff)
        for i in range(num_cols):
            if ipDiff[i] != 0:
                noctimg[:medline[i], i] = (octimg[:medline[i], i] - minIP[i]) / ipDiff[i]
    elif 'ipnonlin' in mode:
        for i in range(num_cols):
            if ipDiff[i] != 0:
                noctimg[:medline[i], i] = ((octimg[:medline[i], i] - minIP[i]) / ipDiff[i]) ** 2
    elif 'ipnearmax' in mode:
        minIP[:] = np.min(minIP)
        maxIPS = np.sort(maxIP)
        start_idx = -int(np.ceil(len(maxIP) / 5))
        end_idx = -int(np.ceil(len(maxIP) / 10))
        maxIP[:] = np.mean(maxIPS[start_idx:end_idx])
        ipDiff = maxIP - minIP
        print("maxIP after 'ipnearmax':", maxIP)
        print("ipDiff after 'ipnearmax':", ipDiff)
        for i in range(num_cols):
            if ipDiff[i] != 0:
                noctimg[:medline[i], i] = (octimg[:medline[i], i] - minIP[i]) / ipDiff[i]
    else:
        # ipsimple mode
        for i in range(num_cols):
            if (ipDiff[i] != 0).all():
                noctimg[:medline[i], i] = (octimg[:medline[i], i] - minIP[i]) / ipDiff[i]

    # Handle opDiff with a minimum threshold
    opDiff = maxOP - minOP
    min_op_diff = 1e-3  # Set a sensible minimum
    opDiff = np.where(opDiff < min_op_diff, min_op_diff, opDiff)
    print("opDiff min:", opDiff.min(), "max:", opDiff.max())

    if 'opbscan' in mode:
        minOP[:] = np.min(minOP)
        maxOP[:] = np.max(minOP)
        opDiff = maxOP - minOP
        print("minOP after 'opbscan':", minOP)
        print("maxOP after 'opbscan':", maxOP)
        print("opDiff after 'opbscan':", opDiff)
        for i in range(num_cols):
            if opDiff[i] != 0:
                noctimg[medline[i]:, i] = (octimg[medline[i]:, i] - minOP[i]) / opDiff[i]
    elif 'opnonlin' in mode:
        for i in range(num_cols):
            if opDiff[i] != 0:
                noctimg[medline[i]:, i] = ((octimg[medline[i]:, i] - minOP[i]) / opDiff[i]) ** 2
    else:
        # opsimple mode
        for i in range(num_cols):
            if opDiff[i] != 0:
                noctimg[medline[i]:, i] = (octimg[medline[i]:, i] - minOP[i]) / opDiff[i]

    noctimg = np.clip(noctimg, 0, 1)
    print('noctimg min:', noctimg.min(), 'max:', noctimg.max())
    print("noctimg statistics:")
    print("min:", noctimg.min())
    print("max:", noctimg.max())
    print("mean:", noctimg.mean())
    print("std:", noctimg.std())

    for i in range(num_cols):
        inner = noctimg[:medline[i], i]
        outer = noctimg[medline[i]:, i]
        if np.any(inner > 0):
            cv2.imwrite(f"./output_intermediates/segments/inner_segment_col_{i}.png", (inner * 255).astype(np.uint8))
        if np.any(outer > 0):
            cv2.imwrite(f"./output_intermediates/segments/outer_segment_col_{i}.png", (outer * 255).astype(np.uint8))    
    return noctimg, medline

def remove_bias(octimg, params):
    """
    Removes bias from the OCT image.

    Parameters:
    octimg (np.ndarray): BScan image.
    params (dict): Parameters with keys 'REMOVEBIAS_REGIONWIDTH' and 'REMOVEBIAS_FRACTION'.

    Returns:
    np.ndarray: Bias-corrected and normalized BScan image.
    """
    if isinstance(octimg, tuple):
        try:
            octimg = np.vstack(octimg)
        except ValueError:
            raise ValueError("All elements in the tuple must have the same shape to stack.")
    if octimg.ndim != 2:
        raise ValueError(f"octimg must be a 2D array, but got {octimg.ndim}D array.")

    regionwidth = params['REMOVEBIAS_REGIONWIDTH']
    fraction = params['REMOVEBIAS_FRACTION']

    # Check if regionwidth exceeds the number of rows
    if regionwidth > octimg.shape[0]:
        raise IndexError(f"regionwidth ({regionwidth}) exceeds number of rows ({octimg.shape[0]}) in octimg.")

    # Extract top and bottom regions
    top_region = octimg[:regionwidth, :].flatten()
    bottom_region = octimg[-regionwidth:, :].flatten()

    # Concatenate and sort
    temp = np.concatenate((top_region, bottom_region))
    temp_sorted = np.sort(temp)

    # Correct bias calculation: Mean of the lower fraction
    index = int(len(temp_sorted) * fraction)
    if index == 0:
        bias = temp_sorted[0]
    else:
        bias = np.mean(temp_sorted[:index])

    # Remove bias and clip
    resoctimg = octimg - bias
    resoctimg = np.clip(resoctimg, 0, 1)

    # Normalize
    max_val = resoctimg.max()
    if max_val != 0:
        resoctimg = resoctimg / max_val
    else:
        resoctimg = resoctimg

    return resoctimg

def extend_blood_vessels(bv, add_width, mult_width_thresh, mult_width):
    """
    Extends blood vessels by a constant factor or multiplicative factor.

    Parameters:
    bv (np.ndarray): Binary array of blood vessel indices.
    add_width (int): Number of positions to add on each side.
    mult_width_thresh (int): Threshold for applying multiplicative extension.
    mult_width (float): Multiplicative factor for extending vessel width.

    Returns:
    np.ndarray: Extended blood vessel indices.
    """
    line_width = bv.size

    # Extend by a constant width
    for _ in range(add_width):
        bv[:-1] = bv[:-1] | bv[1:]
        bv[1:] = bv[1:] | bv[:-1]

    # Extend by a multiplicative factor
    if mult_width != 0:
        bv_new = np.zeros_like(bv)
        j = 0
        while j < line_width:
            if bv[j] == 1:
                a = j
                while a < line_width and bv[a] == 1:
                    a += 1
                length = a - j
                if length > mult_width_thresh:
                    start = int(np.floor(j - mult_width * length))
                    start = max(start, 0)
                    end = int(np.ceil(a + mult_width * length))
                    end = min(end, line_width)
                    bv_new[start:end] = 1
                j = a
            j += 1
        bv = bv_new

    return bv

def find_blood_vessels(bscan, params, linerpe):
    """
    Finds the indices of blood vessel shadows along a line using adaptive thresholding.

    Parameters:
    bscan (np.ndarray): An OCT bscan image.
    params (dict): Parameter dictionary for segmentation.
    linerpe (np.ndarray): The RPE line.

    Returns:
    np.ndarray: Indices of detected blood vessels.
    """
    PRESICION = np.float32  # Equivalent to MATLAB 'single'
    
    multWidthTresh = params['FINDBLOODVESSELS_MULTWIDTHTHRESH']
    multWidth = params['FINDBLOODVESSELS_MULTWIDTH']
    addWidth = params['FINDBLOODVESSELS_FREEWIDTH']
    threshold = params['FINDBLOODVESSELS_THRESHOLD']
    width = params['FINDBLOODVESSELS_WINDOWWIDTH']
    height = params['FINDBLOODVESSELS_WINDOWHEIGHT']
    
    idx = np.zeros(linerpe.shape[1], dtype=np.uint8)
    sumline = np.zeros(linerpe.shape[1], dtype=PRESICION)
    
    # Clamp linerpe values
    linerpe = np.clip(linerpe, height + 1, bscan.shape[0])
    
    for j in range(linerpe.shape[0]):
        # Ensure linerpe[j] is a scalar
        if isinstance(linerpe[j], np.ndarray):
            value = linerpe[j][0] if linerpe[j].size > 0 else 0
        else:
            value = linerpe[j]
        start = int(np.floor(value)) - height
        end = int(np.floor(value)) + 1
        sumline[j] = np.sum(bscan[start:end, j])
    
    # Mirror sumline for padding
    pad = sumline[1:width+1][::-1]
    sumline_padded = np.concatenate((pad, sumline, sumline[-width-1:-1][::-1]))
    
    for j in range(linerpe.shape[1]):
        window = sumline_padded[j:j + 2 * width + 1]
        maxmean = np.mean(window)
        if sumline_padded[j + width] < maxmean * threshold:
            idx[j] = 1
    
    if addWidth != 0 or multWidth != 0:
        idx = extend_blood_vessels(idx, addWidth, multWidthTresh, multWidth)
    
    return np.where(idx > 0)[0]
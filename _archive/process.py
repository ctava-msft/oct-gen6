import matlab.engine
import os

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative paths to the necessary directories
main_path = os.path.join(current_dir, 'octsegSource', 'gui')
io_path = os.path.join(current_dir, 'octsegSource', 'io')
utils_path = os.path.join(current_dir, 'octsegSource', 'utils')

# Verify if the paths exist
if not os.path.exists(main_path):
    raise FileNotFoundError(f"Path does not exist: {main_path}")
if not os.path.exists(io_path):
    raise FileNotFoundError(f"Path does not exist: {io_path}")
if not os.path.exists(utils_path):
    raise FileNotFoundError(f"Path does not exist: {utils_path}")

# Add the relative paths to MATLAB path
eng.addpath(main_path, nargout=0)
eng.addpath(io_path, nargout=0)
eng.addpath(utils_path, nargout=0)

# Call the function to perform all segmentations
eng.perform_all_segmentations(nargout=0)

# Wait for user input before closing the MATLAB engine
input("Press Enter to close the MATLAB engine and exit...")

# Stop MATLAB engine
eng.quit()
import matlab.engine
import os

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative paths to the necessary directories
main_path = os.path.join(current_dir, 'octsegSource', 'gui')
utils_path = os.path.join(current_dir, 'octsegSource', 'io')

# Verify if the paths exist
if not os.path.exists(main_path):
    raise FileNotFoundError(f"Path does not exist: {main_path}")
if not os.path.exists(utils_path):
    raise FileNotFoundError(f"Path does not exist: {utils_path}")

# Add the relative paths to MATLAB path
eng.addpath(main_path, nargout=0)
eng.addpath(utils_path, nargout=0)

# Verify if loadParameters.m exists in the utils_path
load_parameters_path = os.path.join(utils_path, 'loadParameters.m')
if not os.path.exists(load_parameters_path):
    raise FileNotFoundError(f"File does not exist: {load_parameters_path}")

# Call octsegMain.m
eng.octsegMain(nargout=0)

# Wait for user input before closing the MATLAB engine
input("Press Enter to close the MATLAB engine and exit...")

# Stop MATLAB engine
eng.quit()
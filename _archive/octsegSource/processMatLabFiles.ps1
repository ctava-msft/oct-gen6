# Function to process MATLAB files
function Process-Files {
    param (
        [string]$directory
    )

    Get-ChildItem -Path $directory -Filter *.m | ForEach-Object {
        Write-Host "Processing $($_.FullName)"
        #smop $_.FullName
        # Call the Python conversion script
        python convert_matlab_to_python.py "$($_.FullName)"
    }
}

# Get all directories and process MATLAB files in each
Get-ChildItem -Path . -Recurse -Directory | ForEach-Object {
    Process-Files -directory $_.FullName
}

# Process MATLAB files in the current directory
Process-Files -directory .
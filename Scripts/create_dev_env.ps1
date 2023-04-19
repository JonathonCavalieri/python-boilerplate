#This Script creates the dev environment, it assumes that python is already installed

#Script Variables
$folder = "env"
$env_name = "boiler-plate-testing"
#create the env folder if it doesnt already exist
Set-Location $PSScriptRoot
Set-Location ..
If(!(test-path -PathType container $folder))
{
    Write-Output "Creating environment folder..."
    New-Item -ItemType Directory -Path $folder
}else{
    Write-Output "Environment folder already exists"
}

set-location $folder
#Create the virtual env if the folder doesnt exist
If(!(test-path -PathType container $env_name))
{
    Write-Output "Creating virtual python environment..."
    python -m venv $env_name
}else{
    Write-Output "Python virtual environment already exists"
}

#Activate python env
set-location "$env_name\Scripts"
.\Activate.ps1
#Install requirements.txt file
if ($env:VIRTUAL_ENV.EndsWith($env_name))
{
    Write-Output "Installing python requirements..."
    set-location ..\..\..\Scripts
    pip install -r requirements.txt
}

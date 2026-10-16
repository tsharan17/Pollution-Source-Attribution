@echo off
rem Opens the PolSo simulation (3D map, Devices, STM32 and Build tabs) in your default browser.
rem The page itself is self-contained; it needs an internet connection only to load the 3D library.
start "" "%~dp0environment_simulation\results\area_viewer.html"

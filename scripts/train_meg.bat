@echo off
setlocal
cd /d "%~dp0.."
rem Run all subjects; pass inter-subject as the first argument for leave-one-out.
set setting=%~1
if "%setting%"=="" set setting=intra-subject
set data_root=%~2
if "%data_root%"=="" set data_root=data
set batch_size=%~3
if "%batch_size%"=="" set batch_size=128
set lr=5e-5
if "%setting%"=="inter-subject" set lr=1e-5
for %%s in (01 02 03 04) do (
    echo Training MEG sub-%%s, %setting%
    python main.py --dataset meg --subjects sub-%%s --exp_setting %setting% --seed 0 --epoch 50 --lr %lr% --data-root "%data_root%" --batch-size %batch_size%
    if errorlevel 1 exit /b 1
)
endlocal

@echo off
rem Copyright Intel Corporation.
rem 
rem This software and the related documents are Intel copyrighted materials,
rem and your use of them is governed by the express license under which they
rem were provided to you (License). Unless the License provides otherwise,
rem you may not use, modify, copy, publish, distribute, disclose or transmit
rem this software or the related documents without Intel's prior written
rem permission.
rem 
rem This software and the related documents are provided as is, with no
rem express or implied warranties, other than those that are expressly stated
rem in the License.

if "%SETVARS_CALL%"=="1" goto :EOF
if "%I_MPI_ROOT%"=="" goto :EOF

REM rem PATH is setup automatically by Conda.

setlocal
call :exclude_path "%INCLUDE%" "%I_MPI_ROOT%\include"
endlocal & set "INCLUDE=%RESULT%"

setlocal
call :exclude_path "%LIB%" "%I_MPI_ROOT%\lib"
endlocal & set "LIB=%RESULT%"

setlocal
call :exclude_path "%PATH%" "%I_MPI_ROOT%\bin\libfabric\utils"
endlocal & set "PATH=%RESULT%"

setlocal
call :exclude_path "%PATH%" "%I_MPI_ROOT%\bin"
endlocal & set "PATH=%RESULT%"

if not "%CONDA_PREFIX%" == "" goto :skip_bin
setlocal
call :exclude_path "%PATH%" "%I_MPI_ROOT%\bin"
endlocal & set "PATH=%RESULT%"
:skip_bin

rem Restore previous I_MPI_ROOT
if defined I_MPI_ROOT_BKP (
    set "I_MPI_ROOT=%I_MPI_ROOT_BKP:;="&rem "%"
    set "I_MPI_ROOT_BKP=%I_MPI_ROOT_BKP:*;=%"
) else (
    set I_MPI_ROOT=
)

goto :EOF

:exclude_path
    set "VAR1=%~1"
    set "VAR2=%~2"
    set RESULT=

    rem Remove semicolon (;) at the end of line, if it exist.
    if "%VAR1:~-1%"==";" set "VAR1=%VAR1:~0,-1%"

    rem Split str on substr by semicolon
    for %%i in ("%VAR1:;=";"%") do ( call :result_append "%%~i" "%VAR2%" )
exit /b 0

:result_append
    if not "%~1" == "%~2" (
        if not "%RESULT%" == "" (
          set "RESULT=%RESULT%;%~1"
        ) else (
          set "RESULT=%~1"
        )
    )
exit /b 0

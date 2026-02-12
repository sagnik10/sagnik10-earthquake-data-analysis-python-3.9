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
Rem Intel(R) MPI Library Build Environment

if "%SETVARS_CALL%"=="1" goto :EOF

if defined I_MPI_ROOT set "I_MPI_ROOT_BKP=%I_MPI_ROOT%;%I_MPI_ROOT_BKP%"
if "%CONDA_PREFIX%" == "" (
    set "I_MPI_ROOT=%~dp0..\..\..\Library"
    set "PATH=%~dp0..\..\..\Library\bin;%PATH%"
) else (
    set "I_MPI_ROOT=%CONDA_PREFIX%\Library"
)
Rem PATH is setup by Conda automatically.
set "LIB=%I_MPI_ROOT%\lib;%LIB%"
set "INCLUDE=%I_MPI_ROOT%\include;%INCLUDE%"

if /i "%I_MPI_OFI_LIBRARY_INTERNAL%"=="0" goto :EOF
if /i "%I_MPI_OFI_LIBRARY_INTERNAL%"=="no" goto :EOF
if /i "%I_MPI_OFI_LIBRARY_INTERNAL%"=="off" goto :EOF
if /i "%I_MPI_OFI_LIBRARY_INTERNAL%"=="disable" goto :EOF

set "PATH=%I_MPI_ROOT%\bin\libfabric\utils;%I_MPI_ROOT%\bin;%PATH%"

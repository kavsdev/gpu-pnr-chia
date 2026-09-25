#!/bin/bash
set -e
export PATH=/usr/local/cuda/bin:/usr/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/work/dreamplace/DREAMPlace/build2
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/work/dreamplace/install2 -DPython_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_CXX_ABI=1 -DCMAKE_CUDA_ARCHITECTURES=8.9
make -j8
make install
echo REBUILD-DONE

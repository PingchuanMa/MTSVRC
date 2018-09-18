cd build
cmake \
-DCMAKE_INSTALL_PREFIX=${HOME}/anaconda3 \
-DCMAKE_PREFIX_PATH=${HOME}/anaconda3 \
-DCMAKE_BUILD_TYPE=RELEASE \
..
make -j4 install
cd ../pytorch
python setup.py install --with-nvvl=../build

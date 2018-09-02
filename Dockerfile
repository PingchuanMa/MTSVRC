ARG CUDA_VERSION=9.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu16.04

ARG FFMPEG_VERSION=4.0
ARG CMAKE_VERSION=3.10.2

# nvcuvid deps
RUN apt-get update --fix-missing && \
    apt-get install -y libx11-6 libxext6
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# install python
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
            aptitude git vim make wget zip zsh pkg-config \
            build-essential checkinstall p7zip-full python-pip \
            python3-pip tmux ffmpeg i7z unrar htop cmake g++  \
            curl libopenblas-dev python-numpy python3-numpy \
            python python-tk idle python-pmw python-imaging \
            libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
            libtbb2 libtbb-dev  libdc1394-22-dev libavcodec-dev  \
            libavformat-dev libswscale-dev libv4l-dev libatlas-base-dev \
            gfortran && \
    apt-get autoremove && \
    apt-get clean && \
    aptitude install -y python-dev python3-dev && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    # update pip and setuptools
    python -m pip install --upgrade pip setuptools

# minimal ffmpeg from source
RUN apt-get install -y yasm wget && \
    cd /tmp && wget -q http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    tar xf ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    rm ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    cd ffmpeg-$FFMPEG_VERSION && \
    ./configure \
      --prefix=/usr/local \
      --disable-static \
      --disable-all \
      --disable-autodetect \
      --disable-iconv \
      --enable-shared \
      --enable-avformat \
      --enable-avcodec \
      --enable-avfilter \
      --enable-protocol=file \
      --enable-demuxer=mov,matroska \
      --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb && \
    make -j8 && make install && \
    cd /tmp && rm -rf ffmpeg-$FFMPEG_VERSION && \
    apt-get purge -y yasm wget && \
    apt-get autoremove -y

# video_reader build deps (pkg-config, Doxygen, recent cmake)
RUN apt-get install -y pkg-config doxygen wget && \
    cd /tmp && \
    export dir=$(echo $CMAKE_VERSION | sed "s/^\([0-9]*\.[0-9]*\).*/v\1/") && \
    wget -q https://cmake.org/files/${dir}/cmake-$CMAKE_VERSION-Linux-x86_64.sh && \
    /bin/sh cmake-$CMAKE_VERSION-Linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-$CMAKE_VERSION-Linux-x86_64.sh && \
    apt-get purge -y wget && \
    apt-get autoremove -y

# nvidia-docker only provides libraries for runtime use, not for
# development, to hack it so we can develop inside a container (not a
# normal or supported practice), we need to make an unversioned
# symlink so gcc can find the library.  Additional, different
# nvidia-docker versions put the lib in different places, so we make
# symlinks for both places.
RUN ln -s /usr/local/nvidia/lib64/libnvcuvid.so.1 /usr/local/lib/libnvcuvid.so && \
    ln -s libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so

# install opencv
# RUN mkdir -p /software && cd /software && \
#     git clone https://github.com/opencv/opencv.git && \
#     git clone https://github.com/opencv/opencv_contrib.git

# RUN cd /software && \
#     mkdir -p /software/opencv/build && cd /software/opencv/build && \
#     cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local \
#     -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
#     -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_TBB=ON -DWITH_DNN=OFF \
#     -DBUILD_opencv_python2=ON -DBUILD_opencv_python3=ON .. && \
#     make -j4 && \
#     make install && \
#     ldconfig && rm -rf /software

RUN rm -rf /var/lib/apt/lists/*

COPY ./torch-0.3.1-cp35-cp35m-linux_x86_64.whl /software/

RUN python -m pip install /software/torch-0.3.1-cp35-cp35m-linux_x86_64.whl && rm -rf /software

RUN python -m pip install cffi torchvision

RUN mkdir /workspace && cd /workspace && git clone https://github.com/Pika7ma/nvvl.git

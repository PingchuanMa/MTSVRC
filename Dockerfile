FROM nvcr.io/nvidia/tensorrt:18.08-py3

ARG FFMPEG_VERSION=3.4.2
ARG CMAKE_VERSION=3.11.1
ENV DEBIAN_FRONTEND noninteractive

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

COPY ./torch-0.4.0-cp35-cp35m-linux_x86_64.whl /software/
RUN python -m pip install /software/torch-0.4.0-cp35-cp35m-linux_x86_64.whl && rm -rf /software
RUN python -m pip install cffi torchvision
COPY ./ffmpeg-$FFMPEG_VERSION.tar.bz2 /tmp

# minimal ffmpeg from source
RUN apt-get install -y \
      yasm \
      libx264-148 libx264-dev \
      libx265-79 libx265-dev \
      pkg-config && \
    cd /tmp && tar xf ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    rm ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    cd ffmpeg-$FFMPEG_VERSION && \
    ./configure \
    --prefix=/usr/local \
    --disable-static --enable-shared \
    --disable-all --disable-autodetect --disable-iconv \
    --enable-avformat --enable-avcodec --enable-avfilter --enable-avdevice \
    --enable-protocol=file \
    --enable-demuxer=mov,matroska,image2 \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb \
    --enable-gpl --enable-libx264 --enable-libx265 --enable-zlib \
    --enable-indev=lavfi \
    --enable-swresample --enable-ffmpeg \
    --enable-swscale --enable-filter=scale,testsrc,crop \
    --enable-muxer=mp4,matroska,image2 \
    --enable-cuvid --enable-nvenc --enable-cuda \
    --enable-decoder=h264,h264_cuvid,hevc,hevc_cuvid,png,mjpeg,rawvideo \
    --enable-encoder=h264_nvenc,hevc_nvenc,libx264,libx265,png,mjpeg \
    --enable-hwaccel=h264_cuvid,hevc_cuvid \
    --enable-parser=h264,hevc,png && \
    make -j8 && make install && \
    ldconfig && \
    cd /tmp && rm -rf ffmpeg-$FFMPEG_VERSION && \
    apt-get remove -y yasm libx264-dev libx265-dev && \
    apt-get auto-remove -y

# video_reader build deps (pkg-config, Doxygen, recent cmake)
RUN apt-get install -y pkg-config doxygen wget && \
    cd /tmp && \
    export dir=$(echo $CMAKE_VERSION | sed "s/^\([0-9]*\.[0-9]*\).*/v\1/") && \
    wget -q https://cmake.org/files/${dir}/cmake-$CMAKE_VERSION-Linux-x86_64.sh && \
    /bin/sh cmake-$CMAKE_VERSION-Linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-$CMAKE_VERSION-Linux-x86_64.sh && \
    apt-get purge -y wget && \
    apt-get autoremove -y

ARG OPENCV_VERSION=3.4.0
ARG OPENCV_CUDA_GENERATION=Auto

# paired down OpenCV build
COPY ./$OPENCV_VERSION.tar.gz /tmp
RUN apt-get install -y wget && \
    cd /tmp && tar xf $OPENCV_VERSION.tar.gz && \
    rm $OPENCV_VERSION.tar.gz

RUN cd /tmp/opencv-$OPENCV_VERSION && \
    mkdir build && cd build && \
    cmake -DCUDA_GENERATION=$OPENCV_CUDA_GENERATION \
      -DCMAKE_BUILD_TYPE=RELEASE \
      $(for m in cudabgsegm cudacodec cudafeatures2d \
      cudafilters cudalegacy cudaoptflow cudaobjdetect \
      cudawarping dnn features2d flann highgui ml \
      objdetect photo python_bindings_generator shape \
      superres ts video videoio; do echo -DBUILD_opencv_$m=OFF; done) \
      $(for f in WEBP TIFF OPENEXR JASPER; do echo -DWITH_$f=OFF; done) \
      .. && \
    make -j8 && make install && \
    ldconfig && \
    cd /tmp && rm -rf opencv-$OPENCV_VERSION && \
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

RUN rm -rf /var/lib/apt/lists/*

RUN python -m pip install scipy

RUN mkdir /nvvl && cd /nvvl && git clone http://gitlab.sz.sensetime.com/mapingchuan/nvvl.git

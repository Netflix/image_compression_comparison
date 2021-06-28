# ffmpeg - http://ffmpeg.org/download.html
#
# From https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
#
# https://hub.docker.com/r/jrottenberg/ffmpeg/
#
#
FROM        ubuntu:18.04 AS base

WORKDIR     /image_compression_comparison

RUN     apt-get -yqq update && \
        apt-get install -yq --no-install-recommends ca-certificates expat libgomp1 && \
        apt-get autoremove -y && \
        apt-get clean -y

FROM base as build

ARG        PKG_CONFIG_PATH=/opt/ffmpeg/lib/pkgconfig
ARG        LD_LIBRARY_PATH=/opt/ffmpeg/lib
ARG        PREFIX=/opt/ffmpeg
ARG        MAKEFLAGS="-j2"

ENV         FFMPEG_VERSION=4.2.1                \
            FDKAAC_VERSION=0.1.5                \
#            LIBASS_VERSION=0.14.0               \
            OGG_VERSION=1.3.2                   \
            OPUS_VERSION=1.2                    \
            OPENJPEG_VERSION=2.3.1              \
#            THEORA_VERSION=1.1.1                \
            VORBIS_VERSION=1.3.5                \
            VPX_VERSION=1.8.1                   \
            WEBP_VERSION=1.0.3                  \
            X264_VERSION=20191006-2245-stable   \
            X265_VERSION=3.2                    \
            XVID_VERSION=1.3.4                  \
            FREETYPE_VERSION=2.5.5              \
            FRIBIDI_VERSION=0.19.7              \
            FONTCONFIG_VERSION=2.12.4           \
            LIBVIDSTAB_VERSION=1.1.0            \
            KVAZAAR_VERSION=1.2.0               \
            AOM_VERSION=v3.1.1                  \
            LIBAVIF_SHA=6235931                 \
            KDU_PACKAGE=KDU805_Demo_Apps_for_Linux-x86-64_200602       \
            HM_VERSION=HM-16.20+SCM-8.8         \
            SRC=/usr/local

ARG         OGG_SHA256SUM="e19ee34711d7af328cb26287f4137e70630e7261b17cbe3cd41011d73a654692  libogg-1.3.2.tar.gz"
ARG         OPUS_SHA256SUM="77db45a87b51578fbc49555ef1b10926179861d854eb2613207dc79d9ec0a9a9  opus-1.2.tar.gz"
ARG         VORBIS_SHA256SUM="6efbcecdd3e5dfbf090341b485da9d176eb250d893e3eb378c428a2db38301ce  libvorbis-1.3.5.tar.gz"
# ARG         THEORA_SHA256SUM="40952956c47811928d1e7922cda3bc1f427eb75680c3c37249c91e949054916b  libtheora-1.1.1.tar.gz"
ARG         XVID_SHA256SUM="4e9fd62728885855bc5007fe1be58df42e5e274497591fec37249e1052ae316f  xvidcore-1.3.4.tar.gz"
ARG         FREETYPE_SHA256SUM="5d03dd76c2171a7601e9ce10551d52d4471cf92cd205948e60289251daddffa8  freetype-2.5.5.tar.gz"
ARG         LIBVIDSTAB_SHA256SUM="14d2a053e56edad4f397be0cb3ef8eb1ec3150404ce99a426c4eb641861dc0bb  v1.1.0.tar.gz"
# ARG         LIBASS_SHA256SUM="8fadf294bf701300d4605e6f1d92929304187fca4b8d8a47889315526adbafd7  0.13.7.tar.gz"
ARG         FRIBIDI_SHA256SUM="3fc96fa9473bd31dcb5500bdf1aa78b337ba13eb8c301e7c28923fea982453a8  0.19.7.tar.gz"


RUN      buildDeps="autoconf \
                    automake \
                    bzip2 \
                    cmake \
                    curl \
                    exuberant-ctags \
                    g++ \
                    gcc \
                    git \
                    gperf \
                    imagemagick \
                    libexpat1-dev \
                    libpng-dev \
                    libjpeg-dev \
                    libssl-dev \
                    libtool \
                    make \
                    ninja-build \
                    patchelf \
                    perl \
                    pkg-config \
                    python3 \
                    python3-pip \
                    re2c \
                    subversion \
                    unzip \
                    wget \
                    yasm \
                    zlib1g-dev" && \
        apt-get -yqq update && \
        apt-get install -yq --no-install-recommends ${buildDeps} && \
        pip3 install matplotlib

# SQLite3 is to store metrics.
RUN apt-get install -y sqlite3 libsqlite3-dev

# Install nasm 2.14
RUN curl -L https://download.videolan.org/contrib/nasm/nasm-2.14.tar.gz | tar xvz && \
    cd nasm-2.14 && \
    ./configure && make -j2 && make install && \
    cd .. && \
    nasm --version

# Install latest ninja
RUN git clone -b release https://github.com/ninja-build/ninja.git && \
    cd ninja/ && \
    python3 ./configure.py && \
    ninja && \
    cp ninja `which ninja` && \
    cd .. && \
    ninja --version

# AVIFENC
RUN mkdir -p /tools && \
    cd /tools && \
    git clone https://github.com/AOMediaCodec/libavif && \
    cd /tools/libavif && \
    git checkout ${LIBAVIF_SHA} && \
    cd /tools/libavif/ext && \
    $SHELL ./aom.cmd && \
    cd /tools/libavif && \
    mkdir build && cd build && \
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DAVIF_CODEC_AOM=ON -DAVIF_LOCAL_AOM=ON -DAVIF_CODEC_DAV1D=OFF -DAVIF_LOCAL_DAV1D=OFF -DAVIF_CODEC_RAV1E=OFF -DAVIF_LOCAL_RAV1E=OFF -DAVIF_CODEC_LIBGAV1=OFF -DAVIF_LOCAL_LIBGAV1=OFF -DAVIF_BUILD_TESTS=1 -DAVIF_BUILD_APPS=1 && \
    ninja

## opencore-amr https://sourceforge.net/projects/opencore-amr/
# RUN \
#        DIR=/tmp/opencore-amr && \
#        mkdir -p ${DIR} && \
#        cd ${DIR} && \
#        curl -sL https://kent.dl.sourceforge.net/project/opencore-amr/opencore-amr/opencore-amr-${OPENCOREAMR_VERSION}.tar.gz | \
#        tar -zx --strip-components=1 && \
#        ./configure --prefix="${PREFIX}" --enable-shared  && \
#        make && \
#        make install && \
#        rm -rf ${DIR}

## x264 http://www.videolan.org/developers/x264.html
RUN \
        DIR=/tmp/x264 && \
        mkdir -p ${DIR} && \
        cd ${DIR} && \
        curl -sL https://download.videolan.org/pub/videolan/x264/snapshots/x264-snapshot-${X264_VERSION}.tar.bz2 | \
        tar -jx --strip-components=1 && \
        ./configure --prefix="${PREFIX}" --enable-shared --enable-pic --disable-cli && \
        make && \
        make install && \
        rm -rf ${DIR}

### x265 http://x265.org/
RUN \
        DIR=/tmp/x265 && \
        mkdir -p ${DIR} && \
        cd ${DIR} && \
        curl -sL https://download.videolan.org/pub/videolan/x265/x265_${X265_VERSION}.tar.gz  | \
        tar -zx && \
        cd x265_${X265_VERSION}/build/linux && \
        sed -i "/-DEXTRA_LIB/ s/$/ -DCMAKE_INSTALL_PREFIX=\${PREFIX}/" multilib.sh && \
        sed -i "/^cmake/ s/$/ -DENABLE_CLI=OFF/" multilib.sh && \
        ./multilib.sh && \
        make -C 8bit install && \
        rm -rf ${DIR}

### libogg https://www.xiph.org/ogg/
RUN \
        DIR=/tmp/ogg && \
        mkdir -p ${DIR} && \
        cd ${DIR} && \
        curl -sLO http://downloads.xiph.org/releases/ogg/libogg-${OGG_VERSION}.tar.gz && \
        echo ${OGG_SHA256SUM} | sha256sum --check && \
        tar -zx --strip-components=1 -f libogg-${OGG_VERSION}.tar.gz && \
        ./configure --prefix="${PREFIX}" --enable-shared  && \
        make && \
        make install && \
        rm -rf ${DIR}

### libopus https://www.opus-codec.org/
RUN \
        DIR=/tmp/opus && \
        mkdir -p ${DIR} && \
        cd ${DIR} && \
        curl -sLO https://archive.mozilla.org/pub/opus/opus-${OPUS_VERSION}.tar.gz && \
        echo ${OPUS_SHA256SUM} | sha256sum --check && \
        tar -zx --strip-components=1 -f opus-${OPUS_VERSION}.tar.gz && \
        autoreconf -fiv && \
        ./configure --prefix="${PREFIX}" --enable-shared && \
        make && \
        make install && \
        rm -rf ${DIR}

### libvorbis https://xiph.org/vorbis/
RUN \
        DIR=/tmp/vorbis && \
        mkdir -p ${DIR} && \
        cd ${DIR} && \
        curl -sLO http://downloads.xiph.org/releases/vorbis/libvorbis-${VORBIS_VERSION}.tar.gz && \
        echo ${VORBIS_SHA256SUM} | sha256sum --check && \
        tar -zx --strip-components=1 -f libvorbis-${VORBIS_VERSION}.tar.gz && \
        ./configure --prefix="${PREFIX}" --with-ogg="${PREFIX}" --enable-shared && \
        make && \
        make install && \
        rm -rf ${DIR}

### libtheora http://www.theora.org/
#RUN \
#        DIR=/tmp/theora && \
#        mkdir -p ${DIR} && \
#        cd ${DIR} && \
#        curl -sLO http://downloads.xiph.org/releases/theora/libtheora-${THEORA_VERSION}.tar.gz && \
#        echo ${THEORA_SHA256SUM} | sha256sum --check && \
#        tar -zx --strip-components=1 -f libtheora-${THEORA_VERSION}.tar.gz && \
#        ./configure --prefix="${PREFIX}" --with-ogg="${PREFIX}" --enable-shared && \
#        make && \
#        make install && \
#        rm -rf ${DIR}

### libvpx https://www.webmproject.org/code/
RUN \
        DIR=/tmp/vpx && \
        mkdir -p ${DIR} && \
        cd ${DIR} && \
        curl -sL https://codeload.github.com/webmproject/libvpx/tar.gz/v${VPX_VERSION} | \
        tar -zx --strip-components=1 && \
        ./configure --prefix="${PREFIX}" --enable-vp8 --enable-vp9 --enable-vp9-highbitdepth --enable-pic --enable-shared \
        --disable-debug --disable-examples --disable-docs --disable-install-bins  && \
        make && \
        make install && \
        rm -rf ${DIR}

### libmp3lame http://lame.sourceforge.net/
# RUN \
#         DIR=/tmp/lame && \
#         mkdir -p ${DIR} && \
#         cd ${DIR} && \
#         curl -sL https://kent.dl.sourceforge.net/project/lame/lame/$(echo ${LAME_VERSION} | sed -e 's/[^0-9]*\([0-9]*\)[.]\([0-9]*\)[.]\([0-9]*\)\([0-9A-Za-z-]*\)/\1.\2/')/lame-${LAME_VERSION}.tar.gz | \
#         tar -zx --strip-components=1 && \
#         ./configure --prefix="${PREFIX}" --bindir="${PREFIX}/bin" --enable-shared --enable-nasm --enable-pic --disable-frontend && \
#         make && \
#         make install && \
#         rm -rf ${DIR}

### xvid https://www.xvid.com/
RUN \
        DIR=/tmp/xvid && \
        mkdir -p ${DIR} && \
        cd ${DIR} && \
        curl -sLO http://downloads.xvid.org/downloads/xvidcore-${XVID_VERSION}.tar.gz && \
        echo ${XVID_SHA256SUM} | sha256sum --check && \
        tar -zx -f xvidcore-${XVID_VERSION}.tar.gz && \
        cd xvidcore/build/generic && \
        ./configure --prefix="${PREFIX}" --bindir="${PREFIX}/bin" --datadir="${DIR}" --enable-shared --enable-shared && \
        make && \
        make install && \
        rm -rf ${DIR}

### fdk-aac https://github.com/mstorsjo/fdk-aac
RUN \
        DIR=/tmp/fdk-aac && \
        mkdir -p ${DIR} && \
        cd ${DIR} && \
        curl -sL https://github.com/mstorsjo/fdk-aac/archive/v${FDKAAC_VERSION}.tar.gz | \
        tar -zx --strip-components=1 && \
        autoreconf -fiv && \
        ./configure --prefix="${PREFIX}" --enable-shared --datadir="${DIR}" && \
        make && \
        make install && \
        rm -rf ${DIR}

## openjpeg https://github.com/uclouvain/openjpeg
RUN \
        DIR=/tmp/openjpeg && \
        mkdir -p ${DIR} && \
        cd ${DIR} && \
        curl -sL https://github.com/uclouvain/openjpeg/archive/v${OPENJPEG_VERSION}.tar.gz | \
        tar -zx --strip-components=1 && \
        cmake -DBUILD_THIRDPARTY:BOOL=ON -DCMAKE_INSTALL_PREFIX="${PREFIX}" . && \
        make && \
        make install && \
        rm -rf ${DIR}

## freetype https://www.freetype.org/
RUN  \
        DIR=/tmp/freetype && \
        mkdir -p ${DIR} && \
        cd ${DIR} && \
        curl -sLO https://download.savannah.gnu.org/releases/freetype/freetype-${FREETYPE_VERSION}.tar.gz && \
        echo ${FREETYPE_SHA256SUM} | sha256sum --check && \
        tar -zx --strip-components=1 -f freetype-${FREETYPE_VERSION}.tar.gz && \
        ./configure --prefix="${PREFIX}" --disable-static --enable-shared && \
        make && \
        make install && \
        rm -rf ${DIR}

## libvstab https://github.com/georgmartius/vid.stab
RUN  \
        DIR=/tmp/vid.stab && \
        mkdir -p ${DIR} && \
        cd ${DIR} && \
        curl -sLO https://github.com/georgmartius/vid.stab/archive/v${LIBVIDSTAB_VERSION}.tar.gz &&\
        echo ${LIBVIDSTAB_SHA256SUM} | sha256sum --check && \
        tar -zx --strip-components=1 -f v${LIBVIDSTAB_VERSION}.tar.gz && \
        cmake -DCMAKE_INSTALL_PREFIX="${PREFIX}" . && \
        make && \
        make install && \
        rm -rf ${DIR}

## fridibi https://www.fribidi.org/
# + https://github.com/fribidi/fribidi/issues/8
RUN  \
        DIR=/tmp/fribidi && \
        mkdir -p ${DIR} && \
        cd ${DIR} && \
        curl -sLO https://github.com/fribidi/fribidi/archive/${FRIBIDI_VERSION}.tar.gz && \
        echo ${FRIBIDI_SHA256SUM} | sha256sum --check && \
        tar -zx --strip-components=1 -f ${FRIBIDI_VERSION}.tar.gz && \
        sed -i 's/^SUBDIRS =.*/SUBDIRS=gen.tab charset lib/' Makefile.am && \
        ./bootstrap --no-config && \
        ./configure -prefix="${PREFIX}" --disable-static --enable-shared && \
        make -j 1 && \
        make install && \
        rm -rf ${DIR}

## fontconfig https://www.freedesktop.org/wiki/Software/fontconfig/
RUN  \
        DIR=/tmp/fontconfig && \
        mkdir -p ${DIR} && \
        cd ${DIR} && \
        curl -sLO https://www.freedesktop.org/software/fontconfig/release/fontconfig-${FONTCONFIG_VERSION}.tar.bz2 &&\
        tar -jx --strip-components=1 -f fontconfig-${FONTCONFIG_VERSION}.tar.bz2 && \
        ./configure -prefix="${PREFIX}" --disable-static --enable-shared && \
        make && \
        make install && \
        rm -rf ${DIR}

## libass https://github.com/libass/libass
#RUN  \
#        DIR=/tmp/libass && \
#        mkdir -p ${DIR} && \
#        cd ${DIR} && \
#        curl -sLO https://github.com/libass/libass/releases/download/${LIBASS_VERSION}/libass-${LIBASS_VERSION}.tar.gz && \
#        echo ${LIBASS_SHA256SUM} | sha256sum --check && \
#        tar -zx --strip-components=1 -f libass-${LIBASS_VERSION}.tar.gz && \
#        ./autogen.sh && \
#        ./configure -prefix="${PREFIX}" --disable-static --enable-shared && \
#        make && \
#        make install && \
#        rm -rf ${DIR}

## kvazaar https://github.com/ultravideo/kvazaar
RUN \
        DIR=/tmp/kvazaar && \
        mkdir -p ${DIR} && \
        cd ${DIR} && \
        curl -sLO https://github.com/ultravideo/kvazaar/archive/v${KVAZAAR_VERSION}.tar.gz &&\
        tar -zx --strip-components=1 -f v${KVAZAAR_VERSION}.tar.gz && \
        ./autogen.sh && \
        ./configure -prefix="${PREFIX}" --disable-static --enable-shared && \
        make && \
        make install && \
        rm -rf ${DIR}


# LIBAOM
RUN mkdir -p /tools && \
    cd /tools && \
    git clone https://aomedia.googlesource.com/aom && \
    cd aom && \
    git checkout tags/${AOM_VERSION} && \
    mkdir _build && cd _build && \
    cmake .. && \
    make install

## HEVC (HM)
RUN \
    mkdir -p /tools && \
    cd /tools && \
    svn checkout https://hevc.hhi.fraunhofer.de/svn/svn_HEVCSoftware/tags/${HM_VERSION} && \
    cd ${HM_VERSION}/build/linux && \
    make release

# VMAF library
RUN \ 
    DIR=/tmp/vmaf && \
    mkdir -p ${DIR} && \
    cd ${DIR} && \
    wget -O vmaf.zip https://github.com/Netflix/vmaf/archive/v1.3.9.zip && \
    unzip vmaf.zip && \
    rm -f vmaf.zip && \
    cd vmaf-1.3.9 && \
    make && \
    make install


# WEBP
RUN mkdir -p /tools && \
    cd /tools && \
    wget -O libwebp.tar.gz https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.0.2-linux-x86-64.tar.gz  && \
    tar xvzf libwebp.tar.gz && \
    rm -f libwebp.tar.gz
# libGL libraries are needed for webp.
RUN apt install -y libglu1 libxi6


## ffmpeg https://ffmpeg.org/
RUN  \
        DIR=/tmp/ffmpeg && mkdir -p ${DIR} && cd ${DIR} && \
        curl -sLO https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.bz2 && \
        tar -jx --strip-components=1 -f ffmpeg-${FFMPEG_VERSION}.tar.bz2

RUN \
        DIR=/tmp/ffmpeg && mkdir -p ${DIR} && cd ${DIR} && \
        ./configure \
        --disable-debug \
        --disable-doc \
        --disable-ffplay \
        --enable-shared \
        --enable-avresample \
#        --enable-libopencore-amrnb \
#        --enable-libopencore-amrwb \
        --enable-gpl \
#        --enable-libass \
        --enable-libfreetype \
        --enable-libvidstab \
#        --enable-libmp3lame \
        --enable-libopenjpeg \
        --enable-libopus \
#        --enable-libtheora \
        --enable-libvorbis \
        --enable-libvpx \
#        --enable-libwebp \
        --enable-libx265 \
        --enable-libxvid \
        --enable-libx264 \
        --enable-nonfree \
        --enable-openssl \
        --enable-libfdk_aac \
        --enable-libkvazaar \
        --enable-postproc \
        --enable-small \
        --enable-version3 \
        --enable-libvmaf \
        --extra-cflags="-I${PREFIX}/include" \
        --extra-ldflags="-L${PREFIX}/lib" \
        --extra-libs=-ldl \
        --prefix="${PREFIX}" && \
        make && \
        make install && \
        make distclean && \
        hash -r && \
        cd tools && \
        make qt-faststart && \
        cp qt-faststart ${PREFIX}/bin

# JPEG
RUN mkdir -p /tools && \
    cd /tools && \
    wget -O jpeg.zip https://jpeg.org/downloads/jpegxt/reference1367abcd89.zip && \
    unzip jpeg.zip -d jpeg && \
    rm -f jpeg.zip && \
    cd jpeg && \
    ./configure && \
    make final

# KAKADU
RUN mkdir -p /tools && \
    cd /tools && \
    wget -O kakadu.zip http://kakadusoftware.com/wp-content/uploads/${KDU_PACKAGE}.zip && \
    unzip kakadu.zip -d kakadu && \
    rm -f kakadu.zip && \
    patchelf --set-rpath '$ORIGIN/' /tools/kakadu/${KDU_PACKAGE}/kdu_compress && \
    patchelf --set-rpath '$ORIGIN/' /tools/kakadu/${KDU_PACKAGE}/kdu_expand && \
    patchelf --set-rpath '$ORIGIN/' /tools/kakadu/${KDU_PACKAGE}/kdu_v_compress && \
    patchelf --set-rpath '$ORIGIN/' /tools/kakadu/${KDU_PACKAGE}/kdu_v_expand


## cleanup
RUN \
        ldd ${PREFIX}/bin/ffmpeg | grep opt/ffmpeg | cut -d ' ' -f 3 | xargs -i cp {} /usr/local/lib/ && \
        cp ${PREFIX}/bin/* /usr/local/bin/ && \
        cp -r ${PREFIX}/share/ffmpeg /usr/local/share/ && \
        LD_LIBRARY_PATH=/usr/local/lib ffmpeg -buildconf

ENV         LD_LIBRARY_PATH=/usr/local/lib

# COPY --from=build /usr/local /usr/local/

# Let's make sure the app built correctly
# Convenient to verify on https://hub.docker.com/r/jrottenberg/ffmpeg/builds/ console output

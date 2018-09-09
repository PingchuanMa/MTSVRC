import cupy

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport uint16_t
from libc.stdint cimport uint8_t


cdef extern from "cuda_runtime.h":

    cdef struct CUstream_st
    ctypedef CUstream_st* cudaStream_t


cdef extern from "PictureSequence.h":

    ctypedef void* PictureSequenceHandle
    
    cdef PictureSequenceHandle nvvl_create_sequence(uint16_t count)

    cdef PictureSequenceHandle nvvl_create_sequence_device(uint16_t count, int device_id)

    cdef void nvvl_set_layer(
        PictureSequenceHandle sequence,
        const NVVL_PicLayer* layer,
        const char* name)

    cdef void nvvl_sequence_stream_wait(
        PictureSequenceHandle sequence,
        cudaStream_t stream)

    cdef void nvvl_free_sequence(PictureSequenceHandle sequence)

    enum NVVL_ScaleMethod:
        ScaleMethod_Nearest
        ScaleMethod_Linear

    enum NVVL_ChromaUpMethod:
        ChromaUpMethod_Linear

    enum NVVL_ColorSpace:
        ColorSpace_RGB
        ColorSpace_YCbCr

    enum NVVL_PicDataType:
        PDT_NONE
        PDT_BYTE
        PDT_HALF
        PDT_FLOAT

    cdef struct stride:
        size_t x
        size_t y
        size_t c
        size_t n

    cdef struct RGB_Pixel:
        float r
        float g
        float b

    cdef struct NVVL_LayerDesc:

        uint16_t count
        uint8_t channels
        uint16_t width
        uint16_t height
        uint16_t crop_x
        uint16_t crop_y
        uint16_t scale_width
        uint16_t scale_height
        uint16_t scale_shorter_side
        bool center_crop
        bool horiz_flip
        bool normalized
        NVVL_ColorSpace color_space
        NVVL_ChromaUpMethod chroma_up_method
        NVVL_ScaleMethod scale_method
        stride stride
        RGB_Pixel mean
        RGB_Pixel std
        uint16_t test_crops

    cdef struct NVVL_PicLayer:

        NVVL_PicDataType type
        NVVL_LayerDesc desc
        const int* index_map
        int index_map_length
        void* data


cdef extern from "VideoLoader.h":

    ctypedef void* VideoLoaderHandle

    cdef VideoLoaderHandle nvvl_create_video_loader(int device_id)
    
    cdef void nvvl_destroy_video_loader(VideoLoaderHandle loader)
    
    cdef int nvvl_frame_count(VideoLoaderHandle loader, const char* filename)
    
    cdef void nvvl_read_sequence(
        VideoLoaderHandle loader, const char* filename,
        int frame, int count, int interval, int key_base)
    
    cdef PictureSequenceHandle nvvl_receive_frames(
        VideoLoaderHandle loader, PictureSequenceHandle sequence);
    
    cdef PictureSequenceHandle nvvl_receive_frames_sync(
        VideoLoaderHandle loader, PictureSequenceHandle sequence);

    cdef struct Size:
        uint16_t width
        uint16_t height

    cdef Size nvvl_video_size(VideoLoaderHandle loader)

    enum LogLevel:
        LogLevel_Debug
        LogLevel_Info
        LogLevel_Warn
        LogLevel_Error
        LogLevel_None

    cdef void nvvl_set_log_level(VideoLoaderHandle loader, LogLevel level)


cdef class NVVLProcessDesc:

    """Wrapper of NVVL ProcessDesc

    Parameters
    ----------
    type : string, optional
        Type of the output, can be one of "float", "half", or "byte"
        (Default: "float")

    width, height : int, optional
        width and height to crop frame to, set to 0 for full frame
        size (Default: 0)

    scale_width, scale_height : int, optional
        width and height to scale image to before cropping, set to 0
        for no scaling (Default: 0)

    normalized : bool, optional
        Normalize all values to [0, 1] instead of [0, 255] (Default: False)

    random_crop : bool, optional
        If True, the origin of the crop is randomly choosen. If False,
        the crop originates at (0, 0).  (Default: False)

    random_flip : bool, optional
        If True, flip the image horizontally before cropping. (Default: False)

    color_space : enum, optional
        Color space to return images in, one of "RGB" or "YCbCr". (Default: RGB)

    index_map : list of ints, optional
        Map from indices into the decoded sequence to indices in this Layer.

        None indicates a 1-to-1 mapping of the frames from sequence to
        layer.

        For examples, To reverse the frames of a 5 frame sequence, set
        index_map to [4, 3, 2, 1, 0].

        An index of -1 indicates that the decoded frame should not
        be used in this layer. For example, to extract just the
        middle frame from a 5 frame sequence, set index_map to
        [-1, -1, 0, -1, -1].

        The returned tensors will be sized to fit the maximum index in
        this array (if it is provided, they will fit the full sequence
        if it is None).

        (Default: None)

    dimension_order : string, optional
        Order of dimensions in the returned tensors. Must contain
        exactly one of 'f', 'c', 'h', and 'w'. 'f' for frames in the
        sequence, 'c' for channel, 'h' for height, 'w' for width, and
        'h'. (Default: "fchw")

    """

    cdef NVVL_LayerDesc processing

    def __init__(self, type="float",
                 width=0, height=0, scale_width=0, scale_height=0,
                 normalized=False, random_crop=False, random_flip=False,
                 color_space="RGB", index_map=None, dimension_order="fchw",
                 center_crop=False, mean=(0., 0., 0.), std=(1., 1., 1.),
                 scale_shorter_side=0, test_crops=1):

        # if shorter side scale is set, ignore other scale
        self.width = width
        self.height = height
        self.scale_width = scale_width
        self.scale_height = scale_height
        self.scale_shorter_side = scale_shorter_side
        self.normalized = normalized

        if test_crops not in [1, 3, 5]:
            raise NotImplementedError("Only 1, 3 and 5 crops are supported "
                                      "while we got {}".format(test_crops))
        self.test_crops = test_crops

        # if center crop is set, ignore other crop
        self.center_crop = center_crop
        self.random_crop = random_crop
        self.random_flip = random_flip

        self.mean = mean
        self.std = std

        if index_map:
            self.index_map = self.ffi.new("int[]", index_map)
            self.count = max(index_map) + 1
            self.index_map_length = len(index_map)
        else:
            self.index_map = self.ffi.NULL
            self.count = 0
            self.index_map_length = 0

        if color_space.lower() == "rgb":
            self.color_space = lib.ColorSpace_RGB
            self.channels = 3
        elif color_space.lower() == "ycbcr":
            self.color_space = lib.ColorSpace_YCbCr
            self.channels = 3
        else:
            raise ValueError("Unknown color space")

        if type == "float":
            self.tensor_type = torch.cuda.FloatTensor
        elif type == "half":
            self.tensor_type = torch.cuda.HalfTensor
        elif type == "byte":
            self.tensor_type = torch.cuda.ByteTensor
        else:
            raise ValueError("Unknown type")

        self.dimension_order = dimension_order


cdef class NVVLVideoLoader:

    """Wrapper of NVVL VideoLoader

    Args:
        device_id (int): Specify the device id used to load a video.
        log_level (str): Logging level which should be either ``'debug'``, ``'info'``, ``'warn'``,
            ``'error'``, or ``'none'``. Logs with levels >= ``log_level`` is shown. The default is ``'warn'``.

    """

    cdef VideoLoaderHandle handle
    cdef int device_id

    def __init__(self, device_id, log_level='warn'):
        self.handle = nvvl_create_video_loader(device_id)
        if log_level == 'debug':
            nvvl_set_log_level(self.handle, LogLevel_Debug)
        elif log_level == 'info':
            nvvl_set_log_level(self.handle, LogLevel_Info)
        elif log_level == 'warn':
            nvvl_set_log_level(self.handle, LogLevel_Warn)
        elif log_level == 'error':
            nvvl_set_log_level(self.handle, LogLevel_Error)
        elif log_level == 'none':
            nvvl_set_log_level(self.handle, LogLevel_None)
        else:
            raise ValueError(
                'log_level should be either \'debug\', \'info\', \'warn\', '
                '\'error\', or \'none\', but {} was given.'.format(log_level))
        self.device_id = device_id

    def __dealloc__(self):
        nvvl_destroy_video_loader(self.handle)

    def frame_count(self, filename):
        return nvvl_frame_count(self.handle, filename.encode('utf-8'))

    def read_sequence(
            self, filename, frame=0, count=None, channels=3,
            scale_height=0, scale_width=0, crop_x=0, crop_y=0,
            crop_height=None, crop_width=None, scale_method='Linear',
            horiz_flip=False, normalized=False, color_space='RGB',
            chroma_up_method='Linear', out=None):
        """Loads the video from disk and returns it as a CuPy ndarray.

        Args:
            filename (str): The path to the video.
            frame (int): The initial frame number of the returned sequence.
                Default is 0.
            count (int): The number of frames of the returned sequence.
                If it is None, whole frames of the video are loaded.
            channels (int): The number of color channels of the video.
                Default is 3.
            scale_height (int): The height of the scaled video.
                Note that scaling is performed before cropping.
                If it is 0 no scaling is performed. Default is 0.
            scale_width (int): The width of the scaled video.
                Note that scaling is performed before cropping.
                If it is 0, no scaling is performed. Default is 0.
            crop_x (int): Location of the crop within the scaled frame.
                Must be set such that crop_y + height <= original height.
                Default is 0.
            crop_y (int): Location of the crop within the scaled frame.
                Must be set such that crop_x + width <= original height.
                Default is 0.
            crop_height (int): The height of cropped region of the video.
                If it is None, no cropping is performed. Default is None.
            crop_width (int): The width of cropped region of the video.
                If it is None, no cropping is performed. Default is None.
            scale_method (str): Scaling method. It should be either of
                'Nearest' or 'Lienar'. Default is 'Linear'.
            horiz_flip (bool): Whether horizontal flipping is performed or not.
                Default is False.
            normalized (bool): If it is True, the values of returned video is
                normalized into [0, 1], otherwise the value range is [0, 255].
                Default is False.
            color_space (str): The color space of the values of returned video.
                It should be either 'RGB' or 'YCbCr'. Default is 'RGB'.
            chroma_up_method (str): How the chroma channels are upscaled from
                yuv 4:2:0 to 4:4:4. It should be 'Linear' currently.
            out (cupy.ndarray): The output array where the video is loaded.
                This is optional, but if it is given, the memory resion of
                ``out`` is used to load the video. It must have the same shape
                and the dtype as the expected output, and its order must be
                C-contiguous.

        """
        frame_count = self.frame_count(filename)
        if count is None:
            count = frame_count
        if count > frame_count:
            raise ValueError(
                'count should be less than the video length ({}) '
                'but {} was given.'.format(frame_count, count))

        cdef PictureSequenceHandle sequence = nvvl_create_sequence(count)
        cdef Size size = nvvl_video_size(self.handle)
        cdef uint16_t width = size.width if crop_width is None else crop_width
        cdef uint16_t height = size.height if crop_height is None else crop_height
        cdef NVVL_PicLayer layer
        cdef string name = 'pixels'.encode('utf-8')

        with cupy.cuda.Stream() as stream:
            if out is None:
                array = cupy.empty(
                    (count, channels, height, width), dtype=cupy.float32)
                array = cupy.ascontiguousarray(array)
            else:
                if out.dtype != cupy.float32:
                    raise ValueError('The dtype of `out` must be float32')
                if out.shape != (count, channels, height, width):
                    raise ValueError(
                        'The shape of `out` must be ({}, {}, {}, {}) '
                        '(actual: {})'.format(
                            count, channels, height, width, out.shape))
                if not out.flags['C_CONTIGUOUS']:
                    raise ValueError('`out` must be a C-contiguous array')
                array = out
            stream.synchronize()

            layer.type = PDT_FLOAT
            layer.data = <void*><size_t>array.data.ptr
            layer.index_map = NULL
            layer.desc.count = count
            layer.desc.channels = channels
            layer.desc.height = height
            layer.desc.width = width
            layer.desc.crop_x = crop_x
            layer.desc.crop_y = crop_y
            layer.desc.scale_height = scale_height
            layer.desc.scale_width = scale_width
            layer.desc.horiz_flip = horiz_flip
            layer.desc.normalized = normalized

            if color_space == 'RGB':
                layer.desc.color_space = ColorSpace_RGB
            elif color_space == 'YCbCr':
                layer.desc.color_space = ColorSpace_YCbCr
            else:
                raise ValueError(
                    'color_space should be either \'RGB\' or \'YCbCr\' '
                    'but {} was given.'.format(color_space))

            if chroma_up_method == 'Linear':
                layer.desc.chroma_up_method = ChromaUpMethod_Linear
            else:
                raise ValueError(
                    'chroma_up_method should be \'Linear\' '
                    'but {} was given.'.format(chroma_up_method))

            if scale_method == 'Nearest':
                layer.desc.scale_method = ScaleMethod_Nearest
            elif scale_method == 'Linear':
                layer.desc.scale_method = ScaleMethod_Linear
            else:
                raise ValueError(
                    'scale_method should either \'Nearest\' or \'Linear\' '
                    'but {} was given.'.format(scale_method))

            layer.desc.stride.x = 1
            layer.desc.stride.y = width * layer.desc.stride.x
            layer.desc.stride.c = layer.desc.stride.y * height
            layer.desc.stride.n = layer.desc.stride.c * channels
            
            stream.record()
            nvvl_set_layer(sequence, &layer, name.c_str())
            nvvl_read_sequence(self.handle, filename.encode('utf-8'), frame, count)
            nvvl_receive_frames(self.handle, sequence)
            nvvl_sequence_stream_wait(
                sequence, <cudaStream_t><size_t>stream.ptr)
            nvvl_free_sequence(sequence)
            return array

import mmap
import ctypes as ct
from ctypes import pythonapi, py_object, Structure, byref, c_size_t, c_int, c_void_p, c_char_p, POINTER

class Py_buffer(Structure):
    _fields_ = [
        ('buf', c_void_p),
        ('obj', py_object),
        ('len', c_size_t),
        ('itemsize', c_size_t),
        ('readonly', c_int),
        ('ndim', c_int),
        ('format', c_char_p),
        ('shape', POINTER(c_size_t)),
        ('strides', POINTER(c_size_t)),
        ('suboffsets', POINTER(c_size_t)),
        ('internal', c_void_p),
    ]

pythonapi.PyObject_GetBuffer.argtypes = [py_object, POINTER(Py_buffer), c_int]
pythonapi.PyObject_GetBuffer.restype = c_int
pythonapi.PyBuffer_Release.argtypes = [POINTER(Py_buffer)]
pythonapi.PyBuffer_Release.restype = None


class MXCSR(ct.c_uint32):
    RESET_VALUE = 0x1f80  # Default power-on state: FTZ and DAZ are cleared
    
def get_buffer_pointer(obj):
    buf_struct = Py_buffer()
    if pythonapi.PyObject_GetBuffer(obj, byref(buf_struct), 0) != 0:
        raise RuntimeError("Failed to get buffer")
    try:
        return buf_struct.buf
    finally:
        pythonapi.PyBuffer_Release(byref(buf_struct))


get_mxcsr_asm = b'\x0f\xae\x1f\xc3'
get_mxcsr_page = mmap.mmap(-1, mmap.PAGESIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
get_mxcsr_page.write(get_mxcsr_asm)
_get_mxcsr = ct.CFUNCTYPE(None, ct.c_void_p)(get_buffer_pointer(get_mxcsr_page))

set_mxcsr_asm = b'\x0f\xae\x17\xc3'
set_mxcsr_page = mmap.mmap(-1, mmap.PAGESIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
set_mxcsr_page.write(set_mxcsr_asm)
_set_mxcsr = ct.CFUNCTYPE(None, ct.c_void_p)(get_buffer_pointer(set_mxcsr_page))


def get_mxcsr() -> int:
    buf = mmap.mmap(-1, 4, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    pointer = get_buffer_pointer(buf)
    _get_mxcsr(pointer)
    return int.from_bytes(buf.read(4), 'little')

def set_mxcsr(val: int):
    buf = mmap.mmap(-1, 4, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    buf.write(val.to_bytes(4, 'little'))
    pointer = get_buffer_pointer(buf)
    _set_mxcsr(pointer)
    

class FTZKiller():
    def __enter__(self):
        self.mxsr = get_mxcsr()
        set_mxcsr(MXCSR.RESET_VALUE)
        return self
    def __exit__(self, exception_type, exception_value, exception_traceback):
        set_mxcsr(self.mxsr)
    
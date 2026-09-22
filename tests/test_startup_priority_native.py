"""Native configuration validation; real-library checks are explicitly enabled."""
import ctypes
import os
import unittest
from unittest.mock import patch
from qwentts_cpp_cpu._binding import QwenLibrary, QtInitParams, QtCpuOptions

@unittest.skipUnless(os.environ.get("QWEN_STARTUP_TEST_LIBRARY"), "explicit candidate library required")
class StartupPriorityValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=QwenLibrary(os.environ["QWEN_STARTUP_TEST_LIBRARY"])
    def rejected(self, value, codec_threads=1, stream_frames=2, expected="QWENTTS_CPU_STARTUP_PRIORITY"):
        params=QtInitParams()
        params.abi_version=5
        params.talker_path=b"/startup-validation-no-model"
        params.codec_path=b"/startup-validation-no-model"
        params.max_batch=1
        options=QtCpuOptions(1,codec_threads,stream_frames,0,0)
        with patch.dict(os.environ, {"QWENTTS_CPU_STARTUP_PRIORITY":value}):
            ctx=self.library._lib.qt_init_cpu_ex(ctypes.byref(params),1,ctypes.byref(options))
        self.assertFalse(ctx)
        self.assertIn(expected,self.library.last_error())
    def test_unknown_mode_is_rejected_before_loading_models(self):
        self.rejected("fastest",expected="must be off or second_chunk")
    @unittest.skipIf(os.name == "nt", "Windows CRT treats an empty environment value as unset")
    def test_empty_mode_is_rejected(self):
        self.rejected("",expected="must be off or second_chunk")
    def test_priority_requires_overlap(self):
        self.rejected("second_chunk",codec_threads=0,expected="requires codec overlap")
    def test_priority_rejects_one_frame_cadence(self):
        self.rejected("second_chunk",stream_frames=1,expected="2/4-frame stream")

if __name__=="__main__":
    unittest.main()

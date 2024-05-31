import numpy as np
import traceback
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_available_providers,
    get_device,
)


class ONNXModel:
    def __init__(self, model, only_cpu=False):
        """
        args:
            model: 模型路径或 bytes
            cpu: 是否使用 CPU
        """
        session_options = SessionOptions()
        session_options.enable_cpu_mem_arena = False
        session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.log_severity_level = 2
        session_options.intra_op_num_threads = 0
        session_options.inter_op_num_threads = 0

        providers = auto_providers(only_cpu)
        self.session = InferenceSession(model, session_options, providers)

        self.input_names = [v.name for v in self.session.get_inputs()]
        self.output_names = [v.name for v in self.session.get_outputs()]

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        input_dict = dict(zip(self.input_names, [input_content]))
        try:
            return self.session.run(self.output_names, input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise Exception(error_info) from e


def auto_providers(cpu=False):
    if cpu:
        return [("CPUExecutionProvider", {})]

    available_providers = get_available_providers()
    providers = []

    if 'CUDAExecutionProvider' in available_providers:
        options = {
            "device_id": 0,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }
        providers.append(("CUDAExecutionProvider", options))

    elif 'DmlExecutionProvider' in available_providers:
        options = {}
        providers.append(("DmlExecutionProvider", options))

    if 'CPUExecutionProvider' in available_providers:
        options = {}
        providers.append(("CPUExecutionProvider", options))

    return providers

if __name__ == '__main__':
    model = ONNXModel(r"G:\models\paddleocr\ch_PP-OCRv4_det_infer.onnx")
    model(np.random.rand(1, 3, 224, 224).astype(np.float32))

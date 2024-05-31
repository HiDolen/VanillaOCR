from VanillaOCR.onnx import ONNXModel
import numpy as np
from VanillaOCR.utils import get_configs
from VanillaOCR.utils import get_char_type


class LabelDecoder:
    def __init__(
        self,
        char_list,
        number=None,
        zh=None,
        en_upper=None,
        en_lower=None,
        ja=None,
        full_width=None,
        half_width=None,
        others=None,
        default_char_state=True,
    ):
        configs = get_configs()['LabelDecoder']

        self.char_list = ["blank"] + char_list + [' ']

        # Filter indices
        char_types = {
            'reserved': True,
            'number': number,
            'zh': zh,
            'en_upper': en_upper,
            'en_lower': en_lower,
            'ja': ja,
            'full_width': full_width,
            'half_width': half_width,
            'others': others,
        }
        char_types = {k: v if v is not None else default_char_state for k, v in char_types.items()}
        # Add reserved characters
        self.indices = []
        for char_type, enabled in char_types.items():
            if enabled:
                indices = self._filter_indices(lambda x: get_char_type(x) == char_type)
                self.indices.extend(indices)
        # Sort
        self.indices = sorted(self.indices)

    def _filter_indices(self, condition):
        return [i for i, char in enumerate(self.char_list) if condition(char)]

    def __call__(self, label_batch, only_number=False):
        """

        args:
            label_batch: numpy 矩阵，[batch, length, num_classes]
        """
        decode_text_list, prob_avg_list = self._decode(label_batch)
        return decode_text_list, prob_avg_list

    def _decode(self, label_batch):
        def decode_labels(labels):
            char_list = []
            prob_list = []
            labels = labels[:, self.indices]
            argmax_index = np.argmax(labels, axis=-1)
            for i, index in enumerate(argmax_index):
                char_index = self.indices[index]
                if self.char_list[char_index] in ignored_tokens:
                    continue
                # if i > 0 and argmax_index[i - 1] == argmax_index[i]:
                #     continue
                char_list.append(self.char_list[char_index])
                prob_list.append(labels[i][index])
            prob_avg = sum(prob_list) / len(prob_list) if prob_list else 0
            return char_list, prob_avg

        ignored_tokens = ['blank']
        text_batch = []
        prob_batch = []
        for batch in label_batch:
            decode_text_list = []
            prob_avg_list = []
            for labels in batch:
                char_list, prob_avg = decode_labels(labels)
                decode_text_list.append(''.join(char_list))
                prob_avg_list.append(prob_avg)
            text_batch.append(decode_text_list)
            prob_batch.append(prob_avg_list)
        return text_batch, prob_batch

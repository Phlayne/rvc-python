import os
import sys
import traceback

# now_dir = os.getcwd()
# sys.path.append(now_dir)
import logging

import numpy as np

from rvc_python.lib.audio import load_audio

logging.getLogger("numba").setLevel(logging.WARNING)

f = None

def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()
    # TODO Check if mutex (multiprocessing lock) is useful here, it is used in the print in preprocess.py

class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        x = load_audio(path, self.fs)
        # p_len = x.shape[0] // self.hop
        if f0_method == "rmvpe":
            if hasattr(self, "model_rmvpe") == False:
                from rvc_python.lib.rmvpe import RMVPE

                print("Loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=False, device="cuda"
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def go(self, paths, f0_method):
        if len(paths) == 0:
            printt("no-f0-todo")
        else:
            printt("todo-f0-%s" % len(paths))
            # n = max(len(paths) // 5, 1) # Up to 5 prints per process
            n = 1
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt("f0ing,now-%s,all-%s,-%s" % (idx, len(paths), inp_path))
                    if (
                        os.path.exists(opt_path1 + ".npy") == True
                        and os.path.exists(opt_path2 + ".npy") == True
                    ):
                        continue
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(
                        opt_path2,
                        featur_pit,
                        allow_pickle=False,
                    )  # nsf
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(
                        opt_path1,
                        coarse_pit,
                        allow_pickle=False,
                    )  # ori
                except:
                    printt("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))

if __name__ == "__main__":
    n_part = int(sys.argv[1])
    i_part = int(sys.argv[2])
    i_gpu = sys.argv[3]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    inp_root = sys.argv[4]
    exp_dir = sys.argv[5]
    f = open(f'{exp_dir}/extract_f0_feature.log', "a+")

    featureInput = FeatureInput()
    paths = []
    inp_root = f'{inp_root}/1_16k_wavs'
    opt_root1 = f'{exp_dir}/2a_f0'
    opt_root2 = f'{exp_dir}/2b-f0nsf'

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = "%s/%s" % (inp_root, name)
        if "spec" in inp_path:
            continue
        opt_path1 = "%s/%s" % (opt_root1, name)
        opt_path2 = "%s/%s" % (opt_root2, name)
        paths.append([inp_path, opt_path1, opt_path2])
    try:
        featureInput.go(paths[i_part::n_part], "rmvpe")
    except:
        printt("f0_all_fail-%s" % (traceback.format_exc()))
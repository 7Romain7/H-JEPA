import torch  # charge la libstdc++ de conda avant sqlite3/icu (import lance)
import os, sys, time, lance, pyarrow as pa
N_FRAG = int(sys.argv[1])
src = lance.dataset("hf://datasets/fracapuano/lewm-reacher/train.lance")
dst = os.path.expanduser("~/psc/swm_home/datasets/reacher-sub/train.lance")
frags = src.get_fragments()[:N_FRAG]
def batches():
    for i, f in enumerate(frags):
        t = time.time()
        for b in f.to_batches(batch_size=2048):
            yield b
        print(f"fragment {i + 1}/{N_FRAG} ({time.time() - t:.0f}s)", flush=True)
reader = pa.RecordBatchReader.from_batches(src.schema, batches())
lance.write_dataset(reader, dst, mode="create", max_rows_per_file=50250)
d = lance.dataset(dst)
ep = d.to_table(columns=["episode_idx"]).column(0).to_numpy()
print("OK rows", d.count_rows(), "episodes", ep.min(), ep.max(), "contigu", bool((ep[1:] >= ep[:-1]).all()))

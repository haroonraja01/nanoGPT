import numpy as np

import matplotlib.pyplot as plt
import matplotlib

x = [64, 128, 256, 512]
y1 = np.load('out-shakespeare-char/results/time_spent_kv_cache_cpu.npz')
y2 = np.load('out-shakespeare-char/results/time_spent_cpu.npz')
y1 = y1['arr_0']
y2 = y2['arr_0']
plt.plot(x, y1.mean(1), '-o', label ='kv cache')
plt.plot(x, y2.mean(1), '-^', label ='without caching')
plt.xticks([64, 128, 256, 512])
plt.xlabel("Sequence length")
plt.ylabel("Runtime (sec)")
plt.legend()
plt.title('Runtime comparison of kv-caching and no-cache sampling')
matplotlib.rcParams.update({'font.size': 24})
# plt.show()
plt.savefig('out-shakespeare-char/results/runtime.png')
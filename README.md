### BSA（Bayesian Segmentation for Anomalies）

作者：庞力铖
邮箱：3522236586@qq.com
GitHub地址：https://github.com/Plc912/BSA-.git

基于贝叶斯方法的时序分割与结构变化型异常识别工具（MCP工具）。

- **算法**: 正态-逆伽马共轭先验下的边际似然，基于对数贝叶斯因子的二分分割（Bayesian Binary Segmentation），对显著结构变化点进行检测。
- **语言/版本**: Python 3.13.5
- **依赖**: 见 `requirements.txt`
- **MCP**: 参考 `deeplog_mcp_server.py` 的封装方式，使用 `fastmcp.FastMCP`

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行MCP服务

```bash
python bsa_mcp_server.py
```

默认以 SSE 方式在 `2264` 端口启动。

### 工具清单

- `bsa_detect_from_series(series, mu0=0, kappa0=1, alpha0=1, beta0=1, min_seg_len=20, bf_threshold=5.0, max_changes=None)`
  - 输入原始数值序列，返回：`change_points`, `bayes_factors`, `segments`
- `bsa_detect_from_file(file_path, delimiter=",", column=0, skip_header=0, ...)`
  - 从文件列读取序列并检测
- `bsa_generate_synthetic(save_path=None, n_samples=1200, seed=42)`
  - 生成测试数据CSV（列名 `value`），并返回真实分割点

### 参数说明（核心）

- **min_seg_len**: 最小分段长度，避免过拟合；太小会造成噪声误检。
- **bf_threshold**: 接受分割的对数贝叶斯因子阈值（nats，默认5.0约等于BF≈148）。可降低以提高灵敏度。
- 先验超参 `mu0, kappa0, alpha0, beta0` 影响边际似然，默认较平坦；可结合业务做经验设定。

### 典型调用（非MCP，直接Python内调用）

```python
from bsa.bocpd import detect_change_points
from bsa.utils import generate_piecewise_gaussian

series, cps_true = generate_piecewise_gaussian(n_samples=1500, seed=0)
res = detect_change_points(series, min_seg_len=30, bf_threshold=5.0)
print(res["change_points"])  # 包含末尾 n
print(res["segments"])       # (start, end)
```

### 备注

- 若需要自定义阈值或更强的鲁棒性，可调大 `min_seg_len`、调高 `bf_threshold`。
- 本工具采用分段高斯假设；若系列明显非高斯，可预处理或换更贴近分布的似然模型。

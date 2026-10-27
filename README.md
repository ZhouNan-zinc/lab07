# 实验七：白膜转写实（纯 CPU 深度控制流水线）

用 Blender 无头渲染白模的 depth / normal / clay 三张控制图，再用 stable-diffusion.cpp
的 SD-Turbo Q4_0 + SD2.1 depth ControlNet 在纯 CPU 上生成写实图，记录采样/VAE/总耗时，
评估几何一致性与画面缺陷。实验计划在 `../plan.md`，报告在 `report/`（不进入源码提交）。

## 环境

- CPU：Intel Core i7-14650HX（16 核 24 线程），内存 15 GiB，无 GPU。
- OS：Ubuntu 24.04（Linux 7.0.0-28-generic），Python 3.12.3。
- Blender：4.5.13 LTS Linux x64，Cycles CPU，无头模式。
- 推理后端：stable-diffusion.cpp `master-1017-be48649+`，CPU 后端，本地编译。
- 主模型：SD-Turbo 单文件 fp32 权重本地两步转 q4_0（UNet 线性 4-bit，3×3 卷积与 VAE/CLIP 保留 f16/f32，文件 2.2 GB）。
- ControlNet：SD2.1 depth `control_v11p_sd21_depth.safetensors` 转 f16 GGUF。
- Python：numpy 1.26.4、Pillow 10.2.0、opencv 4.6.0、scipy 1.11.4、matplotlib 3.6.3。

依赖见 `requirements.txt`；模型、Blender 和 sd.cpp 构建产物都放在本地忽略目录，不提交。

## 运行

```bash
# 1) 生成白模场景（一次性）
env/blender/blender -b -P blender/make_demo_scene.py -- \
    --out assets/demo_white_model.blend --res 512

# 2) 导出 depth / normal / clay（A 为正式基线）
env/blender/blender -b assets/demo_white_model.blend \
    -P blender/export_control_maps.py -- \
    --outdir results/raw/A --name A --res 512 --samples 64 --no-floor-depth

# 3) CPU 生成；正式实验 A-E
python scripts/run_pipeline.py --tag A --res 512 --steps 4 --cfg 1.2 --control-strength 0.9 --no-floor-depth
python scripts/run_pipeline.py --tag B --res 768 --steps 4 --cfg 1.2 --control-strength 0.9 --no-floor-depth
python scripts/run_pipeline.py --tag C --res 512 --steps 8 --cfg 1.2 --control-strength 0.9 --reuse-raw A
python scripts/run_pipeline.py --tag D --res 512 --steps 4 --cfg 1.2 --control-strength 0.5 --reuse-raw A
python scripts/run_pipeline.py --tag E --res 512 --steps 4 --mist-start 4.5 --mist-end 7.5 --no-floor-depth

# 4) 汇总、耗时图、几何诊断图
python scripts/analyze_results.py --tags A,B,C,D,E
```

模型不存在时，先按 `plan.md` 第 4 节的来源下载并转换；命令与 sha256 记录在
`results/metrics/env.json`、`results/metrics/model_hashes.json`。

## 输出

```text
results/raw/<tag>/             depth / normal / clay / meta.json
results/images/<tag>_photoreal.png
results/logs/<tag>_blender.log、<tag>_sdcli.log
results/metrics/run_<tag>.json、summary.json、summary.csv、env.json
results/figures/               报告用对比/耗时/几何图
```

每次运行保存命令、配置、种子、模型 sha256、耗时、退出码；`summary.json` 由
`scripts/analyze_results.py` 从 run JSON 和图片重新计算，报告数字不手抄。

## 实验组与结果

| 运行 | 分辨率 | 步数 | CFG | ControlNet | 种子 | 采样 (s) | VAE (s) | 总耗时 (s) |
|---|---|---|---|---|---|---|---|---|
| A | 512² | 4 | 1.2 | 0.9 | 42 | 33.08 | 17.47 | 63.23 |
| B | 768² | 4 | 1.2 | 0.9 | 42 | 92.03 | 36.17 | 153.80 |
| C | 512² | 8 | 1.2 | 0.9 | 42 | 66.15 | 16.33 | 89.60 |
| D | 512² | 4 | 1.2 | 0.5 | 42 | 30.35 | 16.21 | 53.90 |
| E | 512² | 4 | 1.2 | 0.9 | 42 | 29.56 | 16.21 | 58.35 |

C/D 复用 A 的控制图，总耗时不含 Blender；A/B/E 含约 5~10 s Blender。E 是 mist 4.5~7.5 m
过小的失败实验，生成图只剩圆柱和立方体。A2 与 A 图像逐字节一致，总耗时 56.87 s（时间差 10.0%），
说明固定种子结果确定、耗时受热降频影响。A 是质量/耗时平衡最好的方案；B 边缘指标略好但立方体顶面圆化；
C/D 明显退化。最终结论与缺陷分析见报告第七、八节。

## 拓展：头像融合

`scripts/compose_head.py` 实现白底软抠图、椭圆约束、预乘 alpha 缩放、亮度对齐、
接触阴影、Poisson 融合和颗粒匹配。示例：

```bash
python scripts/compose_head.py --base results/images/A_photoreal.png \
    --head portrait.png --depth results/raw/A/A_depth.png \
    --out results/images/A_face.png
```

真实人像照片不在仓库内；报告只记录实际做过的合成测试，不伪造身份或肖像。

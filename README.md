# 随动节拍器与伴奏处理实验

## 功能

拍点分析、节拍网格拟合、点击音合成和音频处理脚本。

## 技术栈

Python、NumPy、SciPy、librosa、soundfile、FFmpeg。依赖版本以现有清单和上游兼容要求为准，早期代码可能需要历史版本。

## 目录

caravan_work/build.py 合成；caravan_work/analysis/ 分析；caravan_work/verify.py 检查；encode.sh 编码。

## 启动

在本仓库目录执行；Python 示例需先激活自行创建的虚拟环境。

```sh
python3 -m venv .venv
# 激活虚拟环境后
python3 -m pip install numpy scipy librosa soundfile
# 准备自有且获授权的分轨后
python3 caravan_work/build.py
```

## 公开范围与许可

仅包含脱敏文本源码，不含原 Git 历史、私人绝对路径、凭据、用户数据库/聊天日志、未经逐张审阅的原始图片或二进制媒体。未发布资源不在许可证授权范围内。详见 [LICENSE](LICENSE) 与 [ATTRIBUTION.md](ATTRIBUTION.md)。

维护者有权授权的自有实现代码采用 MIT。第三方依赖、参考实现、团队内容、角色设定与未发布素材不由此重新许可；请同时遵守 ATTRIBUTION.md 所列来源边界。

## 验证与限制

不含第三方整曲、分离音轨、封面和输出。原参数对应历史实验，换音频须重新设置；未验证时长、响度、削波、封面流或拍点精度。

本次执行静态语法/JSON 检查和独立密钥扫描；结果记录在 `VALIDATION.md`。静态通过不代表界面、依赖安装、硬件或端到端运行成功。缺失资源须使用自行创建或获得授权的替代品，不能从原个人素材库自动恢复。

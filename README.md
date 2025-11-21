# jupyterlab conda 初始化记录（for qw2 v 3b inference）

[快速初始化](./init)

一句话命令
```bash
curl -sSL https://raw.githubusercontent.com/lexmount/sagemaker-qwen2-5-vl3b-inference-demo/refs/heads/main/init | bash
```

- 创建notebook Storage调到100G（否则conda安装容量会不够）
- 进入notebook后创建terminal
- 清理 conda 环境

```python
conda deactivate
conda env remove --name hf_llm_env
```

- 创建 conda 环境

```python
conda create -n hf_llm_env python=3.12 -y
conda init
```

- 关闭terminal 新建terminal

```python
conda activate hf_llm_env
conda install -c conda-forge sagemaker
pip install ipykernel
pip install git+https://github.com/huggingface/transformers
python -m ipykernel install --user --name hf_llm_env --display-name "HuggingFace LLM (Python 3.12)"
```

- 在notebook刷新页面，选择kernel “HuggingFace LLM (Python 3.10)”
- 将下面的代码分单元格填入

```python
import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
```

```python
try:
	role = sagemaker.get_execution_role()
except ValueError:
	iam = boto3.client('iam')
	role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

hub = {
	'HF_MODEL_ID':'Qwen/Qwen2.5-VL-3B-Instruct',
	'SM_NUM_GPUS': json.dumps(1)
}

huggingface_model = HuggingFaceModel(
	transformers_version='4.49.0',
	pytorch_version='2.6.0',
	py_version='py312',
	env=hub,
	role=role, 
)
```

```python
# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1,
	instance_type="ml.g5.2xlarge",
	container_startup_health_check_timeout=300,
)
```

```python
predictor.predict({
	"inputs": "Hi, what can you help me with?",
})
```

- 注意！清理endpoint，否则持续计费！

```python
predictor.delete_model()
predictor.delete_endpoint()
```
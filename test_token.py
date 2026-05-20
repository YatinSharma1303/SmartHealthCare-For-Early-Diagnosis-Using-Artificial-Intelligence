from huggingface_hub import HfApi

api = HfApi()
user = api.whoami(token="hf_BmquDISDbKcpuRY")
print(user)
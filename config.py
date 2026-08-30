"""Model configuration for OpenAI and OpenAI-compatible APIs.

Secrets are read from environment variables. Do not write an API key directly
into this file before committing it to version control.
"""



class Config:
    def __init__(self, model_name: str, base_url: str, api_key: str, epoch: int, dataset_name: str = "weibo", batch_size: int = 32):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.epoch = epoch
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.domain_name = ["科技", "军事", "教育考试","灾难事故","政治"]
        self.test_domain_name = ["医药健康", "财经商业", "文体娱乐", "灾难事故","政治"]






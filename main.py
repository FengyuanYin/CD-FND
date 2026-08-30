import sys
import argparse
from trainner import Trainer
from config import Config
import asyncio



def main():
    print("Misinformation Detection Agent is running...")
    config = Config(
        model_name="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key="",
        epoch=10,
        dataset_name="weibo",
        batch_size=32,
    )
    trainer = Trainer(config=config)
    logs = asyncio.run(trainer.train())
    print("Training completed. Logs:")
    for log in logs:
        print(log)




    
if __name__ == "__main__":
    main()
## prepare

1. model
   
   ```bash
   uvx modelscope download --model="ResembleAI/chatterbox-turbo" --local_dir ./model
   ```

2. install repo
   
   ```
   cd repo/chatterbox
   uv --project ../../ add --editable . --default-index https://mirrors.aliyun.com/pypi/simple/ -vv
   ```

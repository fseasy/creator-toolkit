
## 运行依赖

1. prepare model:

   see section `Env build log`, download the `Qwen/Qwen3-TTS-12Hz-0.6B-Base` to `./model-data/Qwen3-TTS-12Hz-0.6B-Base`

2. prepare speaker

   you can call `create-speaker` from fastapi to create new speaker, it will store at `db/speaker`

   in `private-conf/app/creator-toolkit/qwen3-tts`, it stores some preset speakers that used in some project.


## fastapi 接口

package-path：$ROOT/src/fs_qwen3_tts_server
db-path：$ROOT/db 
示例代码：main.py

1. `create-speaker`: 
   创建声音，给定输入： ref-audio, ref-text, speaker-name
   逻辑：
   - 检查 speaker-name 是否存在，不存在，则在 $db-path/speaker 下创建 $speaker-name 文件夹，作为该 speaker 的数据文件夹 记为 speaker-db
   - 将 ref-audio, ref-text 存入 $speaker-db
   - 参考 main.py 中
     ```
      prompt_items = tts.create_voice_clone_prompt(
          ref_audio=ref_audio_single,
          ref_text=ref_text_single,
          x_vector_only_mode=bool(ref-text),
      )
     ```
     然后将 prompt_items 序列化，存储到 $speaker-db 下， 作为后续的 TTS 的声音输入。

2. `batch-tts`:
   发音。
   输入：speaker-name: str, texts: list[str], languages: list[str] | str, audio_fmt: "wav" | "mp3"
   输入检查、处理: 若 languages 是 list, 保证其和 texts 长度对齐；否则，将其扩展为 list[str]
   逻辑：
   - 在全局构建一个 speaker-name -> prompt_items 的 dict (LRU, max-size=5)，记为 speaker_name2prompt
   - 检查 此 speaker-name 是否在 $speaker_name2prompt 中，若在，直接读取出，否则，将其从 $speaker-db 下反序列化出 prompt, 存入 $speaker_name2prompt, 再返回 prompt
   - 参考 main.py 中：

     ```
     tts.generate_voice_clone(
            text=syn_text_batch,
            language=syn_lang_batch,
            voice_clone_prompt=prompt_items,
            **common_gen_kwargs,
        )
     ```

      输出的音频只需要放入内存，或者放入 tmp 后再读取，最后删掉，不要遗留垃圾。
      输出音频默认是 wav, 如果指定的 audio_fmt 是 mp3, 就转换音频格式为 mp3
    
    - 返回：
      是一个 zip 压缩包：内容如下：
      1. manifest.json: {"items": [{"text": str, "language": str, "audio_name": str }] }
      2. xxx.wav/mp3 (对应到 audio_name),
      ...


代码风格：
- 使用 python 3.14 最近标准；path 使用 pathlib；注释用英语，仅包含必要的
- 转换音频格式用 https://github.com/fseasy/fs-pyutils 里的 `audio_to_mp3_bytes` (基于 ffmpeg)
- 使用 fastapi

## Env build log

1. download model:
  
  qwen3-tts: https://github.com/QwenLM/Qwen3-TTS

  ```bash
  modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --local_dir ./model-data/Qwen3-TTS-12Hz-0.6B-Base
  ```

2. conf setting

  bf16 with flash-atten 2.8.3
  - warning: You are attempting to use Flash Attention 2 without specifying a torch dtype. This might lead to unexpected behaviour
  - 4.082Gi/12.000Gi

  如果用 float16, 直接报错; 大概率是溢出(nan)

  如果用 float32
  7.877Gi/12.000Gi


